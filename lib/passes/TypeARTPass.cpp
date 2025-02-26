// TypeART library
//
// Copyright (c) 2017-2025 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#include "Commandline.h"
#include "TypeARTConfiguration.h"
#include "analysis/MemInstFinder.h"
#include "configuration/Configuration.h"
#include "configuration/EnvironmentConfiguration.h"
#include "configuration/FileConfiguration.h"
#include "configuration/PassBuilderUtil.h"
#include "configuration/PassConfiguration.h"
#include "configuration/TypeARTOptions.h"
#include "instrumentation/MemOpArgCollector.h"
#include "instrumentation/MemOpInstrumentation.h"
#include "instrumentation/TypeARTFunctions.h"
#include "support/ConfigurationBase.h"
#include "support/Logger.h"
#include "support/ModuleDumper.h"
#include "support/Table.h"
#include "support/Util.h"
#include "typegen/TypeGenerator.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <llvm/Support/Error.h>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace llvm {
class BasicBlock;
}  // namespace llvm

using namespace llvm;

extern llvm::cl::OptionCategory typeart_category;

static cl::opt<std::string> cl_typeart_configuration_file(
    "typeart-config", cl::init(""),
    cl::desc(
        "Location of the configuration file to configure the TypeART pass. Commandline arguments are prioritized."),
    cl::cat(typeart_category));

#define DEBUG_TYPE "typeart"

ALWAYS_ENABLED_STATISTIC(NumInstrumentedMallocs, "Number of instrumented mallocs");
ALWAYS_ENABLED_STATISTIC(NumInstrumentedFrees, "Number of instrumented frees");
ALWAYS_ENABLED_STATISTIC(NumInstrumentedAlloca, "Number of instrumented (stack) allocas");
ALWAYS_ENABLED_STATISTIC(NumInstrumentedGlobal, "Number of instrumented globals");

namespace typeart::pass {

std::optional<std::string> get_configuration_file_path() {
  if (!cl_typeart_configuration_file.empty()) {
    LOG_DEBUG("Using cl::opt for config file " << cl_typeart_configuration_file.getValue());
    return cl_typeart_configuration_file.getValue();
  }
  const char* config_file = std::getenv("TYPEART_CONFIG_FILE");
  if (config_file != nullptr) {
    LOG_DEBUG("Using env var for types file " << config_file)
    return std::string{config_file};
  }
  LOG_INFO("No configuration file set.")
  return {};
}

class TypeArtPass : public llvm::PassInfoMixin<TypeArtPass> {
  std::optional<config::TypeARTConfigOptions> pass_opts{std::nullopt};
  std::unique_ptr<config::Configuration> pass_config;

  struct TypeArtFunc {
    const std::string name;
    llvm::Value* f{nullptr};
  };

  TypeArtFunc typeart_alloc{"__typeart_alloc"};
  TypeArtFunc typeart_alloc_global{"__typeart_alloc_global"};
  TypeArtFunc typeart_alloc_stack{"__typeart_alloc_stack"};
  TypeArtFunc typeart_free{"__typeart_free"};
  TypeArtFunc typeart_leave_scope{"__typeart_leave_scope"};

  TypeArtFunc typeart_alloc_omp        = typeart_alloc;
  TypeArtFunc typeart_alloc_stacks_omp = typeart_alloc_stack;
  TypeArtFunc typeart_free_omp         = typeart_free;
  TypeArtFunc typeart_leave_scope_omp  = typeart_leave_scope;

  std::unique_ptr<analysis::MemInstFinder> meminst_finder;
  std::unique_ptr<TypeGenerator> typeManager;
  InstrumentationHelper instrumentation_helper;
  TAFunctions functions;
  std::unique_ptr<InstrumentationContext> instrumentation_context;

  const config::Configuration& configuration() const {
    return *pass_config;
  }

 public:
  TypeArtPass() = default;
  explicit TypeArtPass(config::TypeARTConfigOptions opts) : pass_opts(opts) {
    // LOG_INFO("Created with \n" << opts)
  }

  bool doInitialization(Module& m) {
    auto config_file_path = get_configuration_file_path();

    const auto init = config_file_path.has_value()
                          ? config::TypeARTConfigInit{config_file_path.value()}
                          : config::TypeARTConfigInit{{}, config::TypeARTConfigInit::FileConfigurationMode::Empty};

    auto typeart_config = [&](const auto& init_value) {
      if (init_value.mode == config::TypeARTConfigInit::FileConfigurationMode::Empty) {
        return config::make_typeart_configuration_from_opts(pass_opts.value_or(config::TypeARTConfigOptions{}));
      }
      return config::make_typeart_configuration(init_value);
    }(init);

    if (typeart_config) {
      LOG_INFO("Emitting TypeART configuration content\n" << typeart_config.get()->getOptions())
      pass_config = std::move(*typeart_config);
    } else {
      LOG_FATAL("Could not load TypeART configuration.")
      std::exit(EXIT_FAILURE);
    }

    meminst_finder = analysis::create_finder(configuration());

    const std::string types_file =
        configuration().getValueOr(config::ConfigStdArgs::types, {config::ConfigStdArgValues::types});

    const TypegenImplementation typesgen_parser =
        configuration().getValueOr(config::ConfigStdArgs::typegen, {config::ConfigStdArgValues::typegen});
    typeManager = make_typegen(types_file, typesgen_parser);

    LOG_DEBUG("Propagating type infos.");
    const auto [loaded, error] = typeManager->load();
    if (loaded) {
      LOG_DEBUG("Existing type configuration successfully loaded from " << types_file);
    } else {
      LOG_DEBUG("No valid existing type configuration found: " << types_file << ". Reason: " << error.message());
    }

    instrumentation_helper.setModule(m);
    ModuleData mdata{&m};
    typeManager->registerModule(mdata);

    auto arg_collector =
        std::make_unique<MemOpArgCollector>(configuration(), typeManager.get(), instrumentation_helper);
    // const bool instrument_stack_lifetime = configuration()[config::ConfigStdArgs::stack_lifetime];
    auto mem_instrument = std::make_unique<MemOpInstrumentation>(configuration(), functions, instrumentation_helper);
    instrumentation_context =
        std::make_unique<InstrumentationContext>(std::move(arg_collector), std::move(mem_instrument));

    return true;
  }

  bool doFinalization() {
    /*
     * Persist the accumulated type definition information for this module.
     */
    const std::string types_file = configuration()[config::ConfigStdArgs::types];
    LOG_DEBUG("Writing type file to " << types_file);

    const auto [stored, error] = typeManager->store();
    if (stored) {
      LOG_DEBUG("Success!");
    } else {
      LOG_FATAL("Failed writing type config to " << types_file << ". Reason: " << error.message());
    }

    const bool print_stats = configuration()[config::ConfigStdArgs::stats];
    if (print_stats) {
      auto& out = llvm::errs();
      printStats(out);
    }
    return false;
  }

  void declareInstrumentationFunctions(Module& m) {
    // Remove this return if problems come up during compilation
    if (typeart_alloc_global.f != nullptr && typeart_alloc_stack.f != nullptr && typeart_alloc.f != nullptr &&
        typeart_free.f != nullptr && typeart_leave_scope.f != nullptr) {
      return;
    }

    TAFunctionDeclarator decl(m, instrumentation_helper, functions);

    auto alloc_arg_types      = instrumentation_helper.make_parameters(IType::ptr, IType::type_id, IType::extent);
    auto free_arg_types       = instrumentation_helper.make_parameters(IType::ptr);
    auto leavescope_arg_types = instrumentation_helper.make_parameters(IType::stack_count);

    typeart_alloc.f        = decl.make_function(IFunc::heap, typeart_alloc.name, alloc_arg_types);
    typeart_alloc_stack.f  = decl.make_function(IFunc::stack, typeart_alloc_stack.name, alloc_arg_types);
    typeart_alloc_global.f = decl.make_function(IFunc::global, typeart_alloc_global.name, alloc_arg_types);
    typeart_free.f         = decl.make_function(IFunc::free, typeart_free.name, free_arg_types);
    typeart_leave_scope.f  = decl.make_function(IFunc::scope, typeart_leave_scope.name, leavescope_arg_types);

    typeart_alloc_omp.f = decl.make_function(IFunc::heap_omp, typeart_alloc_omp.name, alloc_arg_types, true);
    typeart_alloc_stacks_omp.f =
        decl.make_function(IFunc::stack_omp, typeart_alloc_stacks_omp.name, alloc_arg_types, true);
    typeart_free_omp.f = decl.make_function(IFunc::free_omp, typeart_free_omp.name, free_arg_types, true);
    typeart_leave_scope_omp.f =
        decl.make_function(IFunc::scope_omp, typeart_leave_scope_omp.name, leavescope_arg_types, true);
  }

  void printStats(llvm::raw_ostream& out) {
    const auto scope_exit_cleanup_counter = llvm::make_scope_exit([&]() {
      NumInstrumentedAlloca  = 0;
      NumInstrumentedFrees   = 0;
      NumInstrumentedGlobal  = 0;
      NumInstrumentedMallocs = 0;
    });
    meminst_finder->printStats(out);

    const auto get_ta_mode = [&]() {
      const bool heap   = configuration()[config::ConfigStdArgs::heap];
      const bool stack  = configuration()[config::ConfigStdArgs::stack];
      const bool global = configuration()[config::ConfigStdArgs::global];

      if (heap) {
        if (stack) {
          return " [Heap & Stack]";
        }
        return " [Heap]";
      }

      if (stack) {
        return " [Stack]";
      }

      if (global) {
        return " [Global]";
      }

      LOG_ERROR("Did not find heap or stack, or combination thereof!");
      assert((heap || stack || global) && "Needs stack, heap, global or combination thereof");
      return " [Unknown]";
    };

    Table stats("TypeArtPass");
    stats.wrap_header_ = true;
    stats.title_ += get_ta_mode();
    stats.put(Row::make("Malloc", NumInstrumentedMallocs.getValue()));
    stats.put(Row::make("Free", NumInstrumentedFrees.getValue()));
    stats.put(Row::make("Alloca", NumInstrumentedAlloca.getValue()));
    stats.put(Row::make("Global", NumInstrumentedGlobal.getValue()));

    std::ostringstream stream;
    stats.print(stream);
    out << stream.str();
  }

  llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&) {
    bool changed{false};
    changed |= doInitialization(m);
    const bool heap = configuration()[config::ConfigStdArgs::heap];  // Must happen after doInit
    dump_module(m, heap ? util::module::ModulePhase::kBase : util::module::ModulePhase::kOpt);
    changed |= runOnModule(m);
    dump_module(m, heap ? util::module::ModulePhase::kHeap : util::module::ModulePhase::kStack);
    changed |= doFinalization();
    return changed ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
  }

  bool runOnModule(llvm::Module& m) {
    meminst_finder->runOnModule(m);
    const bool instrument_global = configuration()[config::ConfigStdArgs::global];
    bool globals_were_instrumented{false};
    if (instrument_global) {
      declareInstrumentationFunctions(m);

      const auto& globalsList = meminst_finder->getModuleGlobals();
      if (!globalsList.empty()) {
        const auto global_count = instrumentation_context->handleGlobal(globalsList);
        NumInstrumentedGlobal += global_count;
        globals_were_instrumented = global_count > 0;
      }
    }

    const auto instrumented_function = llvm::count_if(m.functions(), [&](auto& f) { return runOnFunc(f); }) > 0;
    return instrumented_function || globals_were_instrumented;
  }

  bool runOnFunc(llvm::Function& f) {
    using namespace typeart;

    if (f.isDeclaration() || util::starts_with_any_of(f.getName(), "__typeart")) {
      return false;
    }

    if (!meminst_finder->hasFunctionData(f)) {
      LOG_WARNING("No allocation data could be retrieved for function: " << f.getName());
      return false;
    }

    LOG_DEBUG("Running on function: " << f.getName())

    // FIXME this is required when "PassManagerBuilder::EP_OptimizerLast" is used as the function (constant) pointer are
    // nullpointer/invalidated
    declareInstrumentationFunctions(*f.getParent());

    bool mod{false};
    //  auto& c = f.getContext();
    DataLayout dl(f.getParent());

    const auto& fData   = meminst_finder->getFunctionData(f);
    const auto& mallocs = fData.mallocs;
    const auto& allocas = fData.allocas;
    const auto& frees   = fData.frees;

    const bool instrument_heap  = configuration()[config::ConfigStdArgs::heap];
    const bool instrument_stack = configuration()[config::ConfigStdArgs::stack];

    if (instrument_heap) {
      // instrument collected calls of bb:
      const auto heap_count = instrumentation_context->handleHeap(mallocs);
      const auto free_count = instrumentation_context->handleFree(frees);

      NumInstrumentedMallocs += heap_count;
      NumInstrumentedFrees += free_count;

      mod |= heap_count > 0 || free_count > 0;
    }

    if (instrument_stack) {
      const auto stack_count = instrumentation_context->handleStack(allocas);
      NumInstrumentedAlloca += stack_count;
      mod |= stack_count > 0;
    }

    return mod;
  }
};

class LegacyTypeArtPass : public llvm::ModulePass {
 private:
  TypeArtPass pass_impl_;

 public:
  static char ID;  // NOLINT

  LegacyTypeArtPass() : ModulePass(ID){};

  bool doInitialization(llvm::Module&) override;

  bool runOnModule(llvm::Module& module) override;

  bool doFinalization(llvm::Module&) override;

  ~LegacyTypeArtPass() override = default;
};

bool LegacyTypeArtPass::doInitialization(llvm::Module& m) {
  return pass_impl_.doInitialization(m);
}

bool LegacyTypeArtPass::runOnModule(llvm::Module& module) {
  bool changed{false};
  changed |= pass_impl_.runOnModule(module);
  return changed;
}

bool LegacyTypeArtPass::doFinalization(llvm::Module&) {
  return pass_impl_.doFinalization();
  ;
}

}  // namespace typeart::pass

//.....................
// New PM
//.....................
llvm::PassPluginLibraryInfo getTypeartPassPluginInfo() {
  using namespace llvm;
  return {LLVM_PLUGIN_API_VERSION, "TypeART", LLVM_VERSION_STRING, [](PassBuilder& pass_builder) {
            pass_builder.registerPipelineStartEPCallback([](auto& MPM, OptimizationLevel) {
              auto parameters = typeart::util::pass::parsePassParameters(typeart::config::pass::parse_typeart_config,
                                                                         "typeart<heap;stats>", "typeart");
              if (!parameters) {
                LOG_FATAL("Error parsing heap params: " << parameters.takeError())
                return;
              }
              MPM.addPass(typeart::pass::TypeArtPass(typeart::pass::TypeArtPass(parameters.get())));
            });
            pass_builder.registerOptimizerLastEPCallback([](auto& MPM, OptimizationLevel) {
              auto parameters = typeart::util::pass::parsePassParameters(typeart::config::pass::parse_typeart_config,
                                                                         "typeart<no-heap;stack;stats>", "typeart");
              if (!parameters) {
                LOG_FATAL("Error parsing stack params: " << parameters.takeError())
                return;
              }
              MPM.addPass(typeart::pass::TypeArtPass(typeart::pass::TypeArtPass(parameters.get())));
            });
            pass_builder.registerPipelineParsingCallback(
                [](StringRef name, ModulePassManager& module_pm, ArrayRef<PassBuilder::PipelineElement>) {
                  if (typeart::util::pass::checkParametrizedPassName(name, "typeart")) {
                    auto parameters = typeart::util::pass::parsePassParameters(
                        typeart::config::pass::parse_typeart_config, name, "typeart");
                    if (!parameters) {
                      LOG_FATAL("Error parsing params: " << parameters.takeError())
                      return false;
                    }
                    module_pm.addPass(typeart::pass::TypeArtPass(parameters.get()));
                    return true;
                  }
                  LOG_FATAL("Not a valid parametrized pass name: " << name)
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getTypeartPassPluginInfo();
}

//.....................
// Old PM
//.....................
char typeart::pass::LegacyTypeArtPass::ID = 0;  // NOLINT

static RegisterPass<typeart::pass::LegacyTypeArtPass> x("typeart", "TypeArt Pass");  // NOLINT

ModulePass* createTypeArtPass() {
  return new typeart::pass::LegacyTypeArtPass();
}

extern "C" void AddTypeArtPass(LLVMPassManagerRef pass_manager) {
  unwrap(pass_manager)->add(createTypeArtPass());
}

// RUN: %run %s 2>&1 | %filecheck %s

#include "../../lib/runtime/RuntimeInterface.h"
#include "../../lib/typelib/TypeDatabase.h"

#include <cstdio>

struct Datastruct {
  int start;
  double middle;
  float end;
};

struct Secondstruct {
  int start;
  float end;
};

int main(int argc, char** argv) {
  Datastruct data     = {0};
  Secondstruct data_2 = {0};

  auto [invalid_db, db_not_loaded] = typeart::make_database("random_missing_types.yaml");
  if (db_not_loaded) {
    printf("Test database not loaded.\n");
  }

  auto [database, db_load] = typeart::make_database("types.yaml");
  if (db_load) {
    printf("Error not loaded type file.\n");
  }
  printf("Unknown: %i %i %i\n", database->isUnknown(0), database->isUnknown(257),
         database->isUnknown(TYPEART_UNKNOWN_TYPE));
  printf("Unknown struct name: %s\n", database->getTypeName(1000).c_str());

  const auto register_struct = [&database = database](int id, const std::string& name) {
    typeart::StructTypeInfo struct_data{id, name};
    const auto pre_length = database->getStructList().size();
    database->registerStruct(struct_data);
    const auto post_length = database->getStructList().size();
    printf("Type register: %i %i\n", (database->getTypeName(id) == name), (pre_length == post_length));
  };

  register_struct(1, "invalid_built-in");
  register_struct(TYPEART_UNKNOWN_TYPE, "invalid_unkown");
  register_struct(255, "invalid_already");
  register_struct(1000, "valid_struct");

  auto* info = database->getStructInfo(1);
  if (info != nullptr) {
    printf("Error info should be null.\n");
  }

  auto* info_2 = database->getStructInfo(1000);
  if (info_2 == nullptr) {
    printf("Error info should not be null.\n");
  } else {
    printf("Info name: %s\n", info_2->name.c_str());
  }

  const auto size = database->getTypeSize(1001);
  if (size != 0) {
    printf("Error size should be 0.\n");
  }

  return 0;
}

// CHECK: Test database not loaded.
// CHECK-NOT: Error not loaded type file.
// CHECK: Unknown: 0 0 1
// CHECK: Unknown struct name: typeart_unknown_struct
// CHECK: Type register: 0 1
// CHECK: Type register: 0 1
// CHECK: Type register: 0 1
// CHECK: Type register: 1 0
// CHECK-NOT: Error info should be null.
// CHECK-NOT: Error info should not be null.
// CHECK: Info name: valid_struct
// CHECK-NOT: Error size should be 0.
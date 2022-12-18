#! /usr/bin/env python3
#
# TypeART library
#
# Copyright (c) 2017-2022 TypeART Authors
# Distributed under the BSD 3-Clause license.
# (See accompanying file LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
#
# Project home: https://github.com/tudasc/TypeART
#
# SPDX-License-Identifier: BSD-3-Clause
#
import argparse
import json
import logging
import os
import pathlib
import subprocess
import sys

parser = argparse.ArgumentParser(prog='typeart-ir-viewer')
parser.add_argument('source_file')
parser.add_argument('-w', '--wrapper', default=None, help='TypeART wrapper')
parser.add_argument('-p', '--compiledb', default='', help='Compilation database dir')
parser.add_argument('-d', '--diff-viewer', default='meld', help='Diff viewer')
parser.add_argument('-c', '--clean-ir', action='store_const', const=True, default=False, help='Remove IR files')
parser.add_argument('-s', '--skip-viewer', action='store_const', const=True, default=False,
                    help='Only generate IR files, no diff viewer')
parser.add_argument('-f', '--force-ir', action='store_const', const=True, default=False,
                    help='Always generate IR files')
parser.add_argument('-m', '--mode-skip', default='', choices=['heap', 'opt', 'stack'],
                    help='Skip viewing specified phase')


def fetch_typeart_ir_files(source_path, mode_skip):
    source_dir = source_path.parent
    source_file_no_ext = source_path.stem
    logging.debug(f"Generating ir_file paths for {source_path}. Base name: ({source_file_no_ext})")
    ir_files = [pathlib.Path(source_dir, source_file_no_ext + ext) for ext in ['_heap.ll', '_opt.ll', '_stack.ll']]
    if mode_skip:
        ir_files = list(filter(lambda file: not str(file).endswith(mode_skip + '.ll'), ir_files))
    logging.debug(ir_files)
    return ir_files


def find_compilation_db(source_path):
    source_dir = source_path.parent
    logging.debug(f"Trying to find compile_commands.json at {source_dir}")
    compile_cmd_file = None
    while source_dir.is_dir() and source_dir != pathlib.Path("/"):
        logging.debug(f"Looking in {source_dir}, {source_dir.root}")
        if (source_dir / "compile_commands.json").is_file():
            compile_cmd_file = (source_dir / "compile_commands.json")
            logging.debug(f"Found compile commands at {compile_cmd_file}")
            break
        source_dir = source_dir.parent
    return compile_cmd_file


def fetch_compile_commands(config):
    source_path = config.source_file
    if config.compiledb:
        compilation_db_file = (pathlib.Path(config.compiledb) / "compile_commands.json").resolve().absolute()
        logging.debug(f"Using specified {compilation_db_file}")
    else:
        compilation_db_file = find_compilation_db(source_path)
    if compilation_db_file:
        try:
            with open(compilation_db_file) as json_file:
                cc_json = json.load(json_file)
        except:
            logging.debug(f"Error while reading file {compilation_db_file}")
            return []
        for tu in cc_json:
            if pathlib.Path(tu["file"]) == source_path or (
                    pathlib.Path(tu["directory"]) / tu["file"]).absolute() == source_path:
                if "arguments" in tu:
                    tu_args = tu["arguments"][1:]
                else:
                    tu_args = tu["command"].split(" ")[1:]
                logging.debug(f"Found compile arguments {tu_args}")
                return tu_args
    return []


def diff_viewer(config):
    logging.debug(f"Calling {config.diff_viewer} on {config.ir_files}")
    subprocess.check_call([config.diff_viewer] + config.ir_files)


def make_typeart_ir(config):
    source_path = config.source_file
    file_ext = source_path.suffix
    if config.wrapper:
        logging.debug(f"Using cli wrapper {config.wrapper}")
        typeart_wrapper = config.wrapper
    else:
        if file_ext in ['.cpp', '.cc', '.cxx']:
            typeart_wrapper = 'typeart-mpic++-test'
        elif file_ext == ".c":
            typeart_wrapper = 'typeart-mpicc-test'
        else:
            logging.debug(f"Unknown extension {file_ext} for typeart_wrapper")
            sys.exit(1)
    if not config.wrapper_args:
        config.wrapper_args = fetch_compile_commands(config)
    logging.debug(f"Calling wrapper {typeart_wrapper} with \'{config.wrapper_args}\' on {config.source_file}")
    source_dir = source_path.parent
    env = {
        **os.environ,
        "TYPEART_WRAPPER_EMIT_IR": "1",
        "TYPEART_TYPE_FILE": str(config.types_file),
    }
    subprocess.check_call([typeart_wrapper] + config.wrapper_args + [config.source_file], env=env,
                          cwd=source_dir)


def view_typeart_ir(config):
    if config.force_ir or not all(file.exists() for file in config.ir_files) or config.wrapper_args:
        logging.debug(f"Generating IR files for {config.source_file}")
        make_typeart_ir(config)
    if not config.skip_viewer:
        diff_viewer(config)


def rm_ir_files(config):
    for file in config.ir_files:
        os.remove(file)
    if config.types_file.exists():
        os.remove(config.types_file)


class ViewerConfig:
    def __init__(self, args_v):
        self.wrapper_args = []
        args = " ".join(args_v)
        if args.count(" -- ") > 0:
            self.wrapper_args = args.split(" -- ")[1].split(" ")
            logging.debug(f"User arguments for wrapper \'{self.wrapper_args}\'")
        arg_viewer, _ = parser.parse_known_args(args=args_v, namespace=self)
        self.source_file = pathlib.Path(self.source_file).absolute().resolve()
        self.types_file = pathlib.Path(self.source_file.parent, self.source_file.stem + "-types-ir-viewer.yaml")
        self.ir_files = fetch_typeart_ir_files(self.source_file, self.mode_skip)


def main(args):
    config = ViewerConfig(args)
    logging.debug(f"Viewing {config.source_file} with user args {config.wrapper_args}")
    if config.clean_ir:
        rm_ir_files(config)
    else:
        view_typeart_ir(config)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
    exit(0)

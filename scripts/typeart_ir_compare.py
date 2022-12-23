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
import logging
import pathlib
import subprocess
import sys

import typeart_ir_viewer

parser = argparse.ArgumentParser(prog='typeart-ir-compare')
parser.add_argument('source_file')
parser.add_argument('-d', '--diff-viewer', default='meld', help='Diff viewer')
parser.add_argument('-a', '--wrapper-a', help='TypeART wrapper path A')
parser.add_argument('-b', '--wrapper-b', help='TypeART wrapper path B')
parser.add_argument('-m', '--mode', default='heap', choices=['heap', 'opt', 'stack'],
                    help='View diff of specified phase')
parser.add_argument('-s', '--skip-viewer', action='store_const', const=True, default=False,
                    help='Only generate IR files, no diff viewer')


def fetch_typeart_ir_file(source_path, mode):
    source_dir = source_path.parent
    source_file_no_ext = source_path.stem
    logging.debug(f"Generating ir_file path for {source_path}. Base name: ({source_file_no_ext})")
    ext = '_' + mode + '.ll'
    ir_file = pathlib.Path(source_dir, source_file_no_ext + ext)
    logging.debug(ir_file)
    return ir_file


def diff_viewer(config):
    logging.debug(f"Calling {config.diff_viewer} on {config.ir_files}")
    subprocess.check_call([config.diff_viewer] + config.ir_files)


def make_ir_compare(config):
    ir_file_a = make_ir_file(config, config.wrapper_a, '-a')
    ir_file_b = make_ir_file(config, config.wrapper_b, '-b')
    config.ir_files = [ir_file_a, ir_file_b]
    logging.debug(f"Diffing {config.ir_files}")
    if not config.skip_viewer:
        diff_viewer(config)


def make_ir_file(config, wrapper, name):
    std_args = ['-s', '-f', '-w', str(wrapper), str(config.source_file)]
    ir_file = fetch_typeart_ir_file(config.source_file, config.mode)
    args_a = std_args if (config.wrapper_args is None) else std_args + ['--'] + config.wrapper_args
    logging.debug(f"Calling {wrapper} with args: {args_a}")
    typeart_ir_viewer.main(args_a)
    if ir_file.exists():
        logging.debug(f"Renaming file {ir_file}")
        ir_file = ir_file.rename(str(ir_file) + name)
        typeart_ir_viewer.main(['-c', str(config.source_file)])
    return ir_file


class CompareConfig:
    def __init__(self, args_v):
        self.wrapper_args = None
        args = " ".join(args_v)
        if args.count(" -- ") > 0:
            self.wrapper_args = args.split(" -- ")[1].split(" ")
            logging.debug(f"User arguments for wrapper a/b \'{self.wrapper_args}\'")
            parser.parse_known_args(args=args.split(" -- ")[0].split(" "), namespace=self)
        else:
            parser.parse_known_args(args=args_v, namespace=self)
        self.source_file = pathlib.Path(self.source_file).absolute().resolve()
        self.wrapper_a = pathlib.Path(self.wrapper_a).absolute().resolve()
        self.wrapper_b = pathlib.Path(self.wrapper_b).absolute().resolve()


def main(args):
    config = CompareConfig(args)
    logging.debug(f"Comparing {config.source_file}")
    make_ir_compare(config)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
    exit(0)

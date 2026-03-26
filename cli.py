#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def cmd_redock(args):
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("redock.py")),
        "-protein",
        args.protein,
        "-ligand_orig",
        args.ligand_orig,
        "-ligand_new",
        args.ligand_new,
    ]
    if args.vina:
        cmd.extend(["-vina", args.vina])
    if args.chain:
        cmd.extend(["-chain", args.chain])
    return subprocess.run(cmd).returncode


def cmd_pdb_to_fasta(args):
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("pdb_to_fasta.py")),
        "-i",
        args.input,
        "-id",
        args.pdb_id,
    ]
    if args.chain:
        cmd.extend(["-c", args.chain])
    if args.output:
        cmd.extend(["-o", args.output])
    return subprocess.run(cmd).returncode


def cmd_map4(args):
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("map4.py")),
        *args.extra,
    ]
    return subprocess.run(cmd).returncode


def build_parser():
    parser = argparse.ArgumentParser(
        description="aiddtools command line interface for agent and pipeline usage"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_redock = sub.add_parser("redock", help="Run aiddtools redock.py workflow")
    p_redock.add_argument("--protein", required=True)
    p_redock.add_argument("--ligand-orig", required=True, dest="ligand_orig")
    p_redock.add_argument("--ligand-new", required=True, dest="ligand_new")
    p_redock.add_argument("--vina")
    p_redock.add_argument("--chain")
    p_redock.set_defaults(func=cmd_redock)

    p_fasta = sub.add_parser("pdb-to-fasta", help="Extract FASTA from a PDB file")
    p_fasta.add_argument("--input", required=True)
    p_fasta.add_argument("--pdb-id", required=True, dest="pdb_id")
    p_fasta.add_argument("--chain")
    p_fasta.add_argument("--output")
    p_fasta.set_defaults(func=cmd_pdb_to_fasta)

    p_map4 = sub.add_parser("map4", help="Forward arguments to map4.py")
    p_map4.add_argument("extra", nargs=argparse.REMAINDER)
    p_map4.set_defaults(func=cmd_map4)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

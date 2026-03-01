import argparse
from Bio.PDB import PDBParser, Polypeptide
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

def extract_fasta_from_pdb(pdb_file, pdb_id, chain_id=None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)
    model = structure[0]

    sequences = []
    
    for chain in model:
        if chain_id and chain.id != chain_id:
            continue
        try:
            ppb = Polypeptide.PPBuilder()
            for pp in ppb.build_peptides(chain):
                seq = pp.get_sequence()
                record = SeqRecord(seq, id=f"{pdb_id}_{chain.id}", description="")
                sequences.append(record)
        except Exception as e:
            print(f"Error in chain {chain.id}: {e}")

    return sequences

def main():
    parser = argparse.ArgumentParser(description="Extract FASTA sequence from a PDB file")
    parser.add_argument("-i", "--input", required=True, help="Path to the PDB file")
    parser.add_argument("-id", "--pdb_id", required=True, help="PDB ID or identifier")
    parser.add_argument("-c", "--chain", help="Optional: Chain ID to extract")

    args = parser.parse_args()

    records = extract_fasta_from_pdb(args.input, args.pdb_id, args.chain)
    
    if records:
        output_file = f"{args.pdb_id}.fasta"
        SeqIO.write(records, output_file, "fasta")
        print(f"FASTA sequence written to {output_file}")
    else:
        print("No sequences found.")

if __name__ == "__main__":
    main()

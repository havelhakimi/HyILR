
import tarfile
import os

def create_tar_xz(input_file, output_tar_xz):
    """
    Compress a file into a .tar.xz archive.

    :param input_file: Path to the file or directory to compress.
    :param output_tar_xz: Path for the output .tar.xz archive.
    """
    with tarfile.open(output_tar_xz, "w:xz") as tar:
        tar.add(input_file, arcname=os.path.basename(input_file))
    print(f"Created {output_tar_xz}")

# Example usage:
input_file = "/scratch/gaurav_k.iitr/ashish/HiSLR/data/wos/Y.txt"  # Replace with the file or directory you want to compress
output_tar_xz = "Y.tar.xz"  # Name of the resulting .tar.xz file
create_tar_xz(input_file, output_tar_xz)


"""
import torch
from transformers import AutoTokenizer


tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')

x=torch.load('/scratch/gaurav_k.iitr/ashish/HiSLR/data/bert_value_dict.pt')

y={i:tokenizer.encode(j.strip().lower(), truncation=True)  for i,j in x.items() }

print(y)

torch.save(y,'bert_value_dict.pt')

"""
# -*- encoding:utf-8 -*-
import sys
import torch
import argparse
import collections

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_model_path", type=str, help="The input model.")
    parser.add_argument("--output_model_path", type=str, help="The output model.")
    parser.add_argument("--delete_module_prefix", action="store_true", help="Delete 'module.' prefix.")
    parser.add_argument("--add_module_prefix", action="store_true", help="Add 'module.' prefix.")

    args = parser.parse_args()

    prefix = "module."
    input_model = torch.load(args.input_model_path)
    if args.delete_module_prefix:
        output_model = collections.OrderedDict()
        for k, v in input_model.items():
            if prefix not in k:
                print("This model is already of Single-GPU version. Nothing changed.")
                sys.exit(0)
            else:
                output_model[k[len(prefix):]] = v
        print("A Single-GPU version model is created")
        torch.save(output_model, args.output_model_path)

        
    if args.add_module_prefix:
        output_model = collections.OrderedDict()
        for k, v in input_model.items():
            if prefix in k:
                print("This model is already of Multi-GPU version. Nothing changed.")
                sys.exit(0)
            else:
                output_model[prefix+k] = v
        print("A Multi-GPU version model is created")
        torch.save(output_model, args.output_model_path)

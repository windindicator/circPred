import argparse
import fm
import torch
import model


def get_sequence(path):
    file = open(path, 'r')
    lines = file.readlines()
    sequence = ""
    sequence_name = []
    sequence_list = []
    sequence_name_list = []
    for line in lines:
        if line.startswith('>'):
            if sequence == "":
                continue
            sequence_list.append(sequence)
            sequence_name_list.append(sequence_name)
            sequence_name = line[1:]
            sequence = ""
        else:
            sequence += line.strip('\n')
    return sequence_list, sequence_name_list


def main():
    parser = argparse.ArgumentParser(description="Classification Baseline Inference")
    parser.add_argument(
        "--data_path", default=None, help="path to data file or folder", type=str
    )
    parser.add_argument(
        "--task", default=None, help="model to use", type=str
    )
    args = parser.parse_args()

    model, alphabet = fm.pretrained.rna_fm_t12()
    model = model.cuda()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    results_file = open("result.txt", 'w')
    sequence_list, sequence_name_list = get_sequence(args.data_path)

    if args.task == 'identification':
        model_file = "./pretrained/classification_full_length_model.pth"
        rna_model = git_hub_model.classification_model()
        rna_model.load_state_dict(torch.load(model_file))
        rna_model.cuda()
        for i in range(len(sequence_list)):
            if len(sequence_list[i]) < 1000:
                data = [(0, sequence_list[i])]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.cuda()
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[12])
                token_embeddings = results["representations"][12]
                token_embeddings = torch.squeeze(token_embeddings)
                embedding = torch.mean(token_embeddings, 0)
                embedding = torch.unsqueeze(embedding, 0)
                embedding = torch.unsqueeze(embedding, 0)
                output = rna_model(embedding)
                output = torch.squeeze(output)
                x = torch.argmax(output)
                if x == 1:
                    results_file.write(sequence_name_list[i] + " is a circRNA\n")
                else:
                    results_file.write(sequence_name_list[i] + " is not a circRNA\n")
    if args.task == 'scanning':
        sig = torch.nn.Sigmoid()
        model_file = "./pretrained/test_full_length_model.pth"
        rna_model = git_hub_model.scanning_model()
        rna_model.load_state_dict(torch.load(model_file))
        rna_model.cuda()
        for i in range(len(sequence_list)):
            count = 0
            sequence = sequence_list[i]
            while len(sequence) % 500 > 0:
                sequence += 'N'
                count += 1
            if len(sequence) >= 30000:
                continue
            data = []
            for pos in range(int(len(sequence) / 500) - 2):
                data.append((str(pos), sequence[pos * 500:pos * 500 + 1000]))
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.cuda()
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[12])
            token_embeddings = results["representations"][12]
            total_outputs = torch.zeros(len(sequence))
            total_outputs = total_outputs.cuda()
            for pos in range(token_embeddings.shape[0]):
                embedding = token_embeddings[pos, :, :]
                embedding = embedding[1:1001, :]
                embedding = embedding.T
                embedding = embedding.cuda()
                embedding = torch.unsqueeze(embedding, 0)
                output = rna_model(embedding)
                output = sig(output)
                total_outputs[pos * 500:pos * 500 + 1000] = (total_outputs[pos * 500:pos * 500 + 1000] + output) / 2
            total_outputs = total_outputs[500:len(total_outputs) - 500]
            total_outputs = torch.unsqueeze(total_outputs, 0)
            total_outputs = torch.unsqueeze(total_outputs, 0)
            total_outputs = torch.squeeze(total_outputs)
            total_outputs = torch.round(total_outputs)
            total_outputs = total_outputs[:-count]
            results_file.write(sequence_name_list[i] + "\n")
            results_file.write(sequence_list[i] + "\n")
            results_file.write(total_outputs + "\n")
        if args.task == 'mining':
            model_file = "./pretrained/classification_pre-rna_model.pth"
            rna_model = git_hub_model.classification_model()
            rna_model.load_state_dict(torch.load(model_file))
            rna_model.cuda()
            for i in range(len(sequence_list)):
                count = 0
                sequence = sequence_list[i]
                while len(sequence) % 500 > 0:
                    sequence += 'N'
                    count += 1
                if len(sequence) >= 30000:
                    continue
                data = []
                for pos in range(int(len(sequence) / 500) - 2):
                    data.append((str(pos), sequence[pos * 500:pos * 500 + 1000]))
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.cuda()
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[12])
                token_embeddings = results["representations"][12]
                embedding = torch.mean(token_embeddings, 0)
                embedding = torch.mean(embedding, 0)
                embedding = torch.unsqueeze(embedding, 0)
                embedding = torch.unsqueeze(embedding, 0)
                output = rna_model(embedding)
                output = torch.squeeze(output)
                x = torch.argmax(output)
                if x == 1:
                    results_file.write(sequence_name_list[i] + " is a pre-RNA of circRNA\n")
                else:
                    results_file.write(sequence_name_list[i] + " is not a pre-RNA of circRNA\n")

import click
import torch


from sklearn.model_selection import train_test_split


@click.command()
def main():
    embeddings = torch.load("embeddings/embeddings.pt")
    with open("embeddings/indices.txt") as in_file:
        indices = [row.split("|||")[0] for row in in_file.read().split("\n") if row != ""]

    train_idx, test_idx = train_test_split(range(len(indices)))

    train_embeddings = embeddings[train_idx]
    train_indices = [w for i, w in enumerate(indices) if i in train_idx]

    torch.save(train_embeddings, "embeddings/train_embeddings.pt")
    with open("embeddings/train_indices.txt", "w") as out_file:
        out_file.write("\n".join(train_indices))

    test_embeddings = embeddings[test_idx]
    test_indices = [w for i, w in enumerate(indices) if i in test_idx]

    torch.save(test_embeddings, "embeddings/test_embeddings.pt")
    with open("embeddings/test_indices.txt", "w") as out_file:
        out_file.write("\n".join(test_indices))


if __name__ == "__main__":
    main()

#  LocalWords:  txt

from task_datasets.utils.base_dataset import BaseDataset, KAND_get_loader
from task_datasets.utils.kand_creation import CLIP_KAND_Dataset
from backbones.identity import Identity
from backbones.kandcnn import FFNN
from backbones.disent_encoder_decoder import DecoderConv64
import time, torch
from argparse import Namespace
import os


class CLIPKandinsky(BaseDataset):
    NAME = "clipkandinsky"

    def get_data_loaders(self):
        start = time.time()

        self.dataset_train = CLIP_KAND_Dataset(base_path="data", split="train")
        self.dataset_val = CLIP_KAND_Dataset(base_path="data", split="val")
        self.dataset_test = CLIP_KAND_Dataset(base_path="data", split="test")
        # dataset_ood   = KAND_Dataset(base_path='data/kandinsky/data',split='ood')

        # dataset_train.mask_concepts("red-and-squares")

        print(f"Loaded datasets in {time.time()-start} s.")

        print(
            "Len loaders: \n train:",
            len(self.dataset_train),
            "\n val:",
            len(self.dataset_val),
        )
        print(" len test:", len(self.dataset_test))  # , '\n len ood', len(dataset_ood))

        train_loader = KAND_get_loader(
            self.dataset_train, self.args.batch_size, val_test=False
        )
        val_loader = KAND_get_loader(
            self.dataset_val, self.args.batch_size, val_test=True
        )
        test_loader = KAND_get_loader(
            self.dataset_test, self.args.batch_size, val_test=True
        )

        # self.ood_loader = get_loader(dataset_ood,  self.args.batch_size, val_test=True)

        return train_loader, val_loader, test_loader

    def get_backbone(self, args=None):
        return FFNN().to(torch.float64), None
        # return Identity(6,3), None

    def get_split(self):
        return 3, ()

    def print_stats(self):
        print("Hello CLIP")

    def get_labels(self):
        return [str(i) for i in range(2)]

    def get_concept_labels(self):
        return ["square", "circle", "triangle"], ["red", "yellow", "blue"]

    def _create_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_tcav_kand_loader(self, d_type: str, folder_name="kand-tcav-clip-all-in"):

        # 1 as batch size and not shuffled
        if d_type == "train":
            dataloader = KAND_get_loader(self.dataset_train, 1, val_test=True)
        elif d_type == "val":
            dataloader = KAND_get_loader(self.dataset_val, 1, val_test=True)
        else:
            dataloader = KAND_get_loader(self.dataset_test, 1, val_test=True)

        self._create_dir(f"data/{folder_name}")

        counter = []
        for _ in range(18 * 3):
            counter.append(0)

        limit = 1000

        for i, data in enumerate(dataloader):
            images, _, concepts = data

            # get the list of images
            full_image = images.squeeze(0)

            # concept vector
            concept_vector = concepts.squeeze(0)

            # reached the limit
            if all(x > limit for x in counter):
                break

            # loop over the images
            for c_imgs_idx in range(concept_vector.shape[0]):
                # loop over the geometric figures
                for geom_figure_idx in range(3):
                    # cut off the first elment per each image, only
                    current_concept_shape_color = concept_vector[
                        c_imgs_idx, [geom_figure_idx, geom_figure_idx + 3]
                    ]

                    # compute the offset
                    offset_shape = (
                        18 * c_imgs_idx
                        + 6 * geom_figure_idx
                        + current_concept_shape_color[0]
                    )
                    offset_color = (
                        18 * c_imgs_idx
                        + 6 * geom_figure_idx
                        + 3
                        + current_concept_shape_color[1]
                    )

                    concept_name_shape = ""
                    if current_concept_shape_color[0] == 0:
                        concept_name_shape = "square"
                    elif current_concept_shape_color[0] == 1:
                        concept_name_shape = "triangle"
                    elif current_concept_shape_color[0] == 2:
                        concept_name_shape = "circle"
                    else:
                        raise ValueError("No shape found")

                    concept_name_color = ""
                    if current_concept_shape_color[1] == 0:
                        concept_name_color = "red"
                    elif current_concept_shape_color[1] == 1:
                        concept_name_color = "yellow"
                    elif current_concept_shape_color[1] == 2:
                        concept_name_color = "blue"
                    else:
                        raise ValueError("No color found")

                    # create folder if needed
                    if counter[offset_shape] == 0:
                        self._create_dir(
                            f"data/{folder_name}/{c_imgs_idx}_{geom_figure_idx}_{concept_name_shape}"
                        )

                    if counter[offset_color] == 0:
                        self._create_dir(
                            f"data/{folder_name}/{c_imgs_idx}_{geom_figure_idx}_{concept_name_color}"
                        )

                    # select concept
                    if counter[offset_shape] > limit and counter[offset_color] > limit:
                        continue

                    if not counter[offset_shape] > limit:
                        torch.save(
                            full_image,
                            f"data/{folder_name}/{c_imgs_idx}_{geom_figure_idx}_{concept_name_shape}/{counter[offset_shape]}.pt",
                        )
                        counter[offset_shape] += 1

                    if not counter[offset_color] > limit:
                        torch.save(
                            full_image,
                            f"data/{folder_name}/{c_imgs_idx}_{geom_figure_idx}_{concept_name_color}/{counter[offset_color]}.pt",
                        )
                        counter[offset_color] += 1

        print("Total: ", counter)

        print("Done")


if __name__ == "__main__":
    args = Namespace(
        backbone="neural",
        preprocess=0,
        finetuning=0,
        batch_size=256,
        n_epochs=20,
        validate=1,
        dataset="boia",
        lr=0.001,
        exp_decay=0.99,
        warmup_steps=1,
        wandb=None,
        task="boia",
        boia_model="ce",
        model="sddoiadpl",
        c_sup=0,
        which_c=-1,
    )

    dataset = CLIPKandinsky(args)

    train, val, test = dataset.get_data_loaders()
    dataset.save_tcav_kand_loader("val")

from types import SimpleNamespace
from typing import Optional
import torch
from torch import nn
import pyvene
from models.mnistdpl import MnistDPL
from models.mnistnn import MnistNN


class WrappedMnistDPL(nn.Module):
    """
    The WrappedMnistDPL class wraps an instance of the MnistDPL model to make it compatible with pyvene’s intervention framework.
    ConceptExtractionBlock (Nested Class):
    - Acts as a wrapper around the encoder portion of the model.
    - Since pyvene can only “hook” into nn.Module objects, reimplementing this part allows the framework to access and intervene
      on the intermediate (concept) representations.
    Pyvene Configuration:
    - The wrapper sets up a simple configuration and registers a mapping in type_to_module_mapping and type_to_dimension_mapping.
    - Pyvene knows which layer to intervene on, the target is the concept_layer output of the ConceptExtractionBlock.
    """

    class ConceptExtractionBlock(nn.Module):
        """
        A wrapper block that reimplements the behavior of the wrapped model for the concept extraction part.
        This block is needed to make the intermediate computations accessible, as pyvene can only hock onto nn.Modules.
        """

        def __init__(self, encoder, n_images):
            super().__init__()
            self.encoder = encoder
            self.n_images = n_images

        def forward(self, x):
            """
            Reimplementation of MnistDPL forward method to allow pyvene to interact with the desired concept layer.
            Args:
                x: image tensor of size [bs, 1, 28, 56]

            Returns: This module outputs the hidden concept layer [bs, 1, 20] that will be rotated by DAS.
            """
            cs = []
            xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)  # splits images into "n_images" digits
            for i in range(self.n_images):
                # xs[i] is always [bs, 1, 28, 56]
                lc, _, _ = self.encoder(xs[i])  # encodes each sub image xs into its latent representation lc [bs, 2/n_images, 10]
                cs.append(lc)
            # cs is a list of len n_images containing tensor([bs, 2/n_images, 10])
            concept_layer = torch.cat(cs, dim=1).view(x.size(0), 1, -1)  # Ensure final shape [bs, 1, 20] as expected by pyvene
            return concept_layer

    def __init__(self, wrapped_model: MnistDPL):
        super().__init__()
        self.wrapped_model = wrapped_model
        self.device = wrapped_model.device
        self.h_dim = int(2 * self.wrapped_model.n_facts)  # We define it as 10 * 2 = 20
        assert self.h_dim % 2 == 0

        # Pyvene expects self.config
        self.config = SimpleNamespace(
            num_classes=19,
            n_layer=2,
            h_dim=self.h_dim,  # Dimension of the concept layer
            pdrop=0.0,
            problem_type="classification",
            squeeze_output=True,
            include_emb=False,
            include_bias=False
        )

        # Define the layer we want pyvene to intervene on
        self.concept_layer = self.ConceptExtractionBlock(self.wrapped_model.encoder, self.wrapped_model.n_images)

        # Define the module and dimensionality for intervention mapping with pyvene global vars
        pyvene.type_to_module_mapping[type(self)] = {
            "block_output": ("concept_layer", pyvene.models.constants.CONST_OUTPUT_HOOK),
        }  # pyvene expects either a nn.Modul (e.g. "concept_layer") or a nn.ModuleList (e.g. "h[%s]")
        pyvene.type_to_dimension_mapping[type(self)] = {
            "block_output": ("h_dim",),
        }  # pyvene expects the dimensionality of the module to be: [batch_size, 1, h_dim]

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        # Choose the input: if input_ids is provided, use it; otherwise, try inputs_embeds.
        if input_ids is not None:
            x = input_ids
        elif inputs_embeds is not None:
            x = inputs_embeds
        else:
            raise ValueError("No input provided. Please supply either input_ids or inputs_embeds.")
        concept_layer = self.concept_layer(x)  # returns the potentially intervened concepts

        cs = concept_layer.view(concept_layer.size(0), 2, self.wrapped_model.n_facts)
        # Recalculate cs from concept_layer so that it is again of shape [bs, 2, 10]
        # Since concept_layer is [bs, 1, 20] and 1*20 equals 2*10, we can reshape it.

        pCs = self.wrapped_model.normalize_concepts(cs)
        # applies softmax on cs to ensure they represent probability distributions over the possible digit values (0-9)

        # Problog inference to compute worlds and query probability distributions
        py, _ = self.wrapped_model.problog_inference(pCs)

        logits = py
        return (logits,)


class WrappedMnistNN(nn.Module):
    """
    Wraps an instance of the MnistNN model to make it compatible with pyvene’s intervention framework.
    """

    class ConceptExtractionBlock(nn.Module):
        """
        A wrapper block that reimplements the behavior of the wrapped model for the concept extraction part.
        This block is needed to make the intermediate computations accessible, as pyvene can only hock onto nn.Modules.
        """

        def __init__(self, encoder, n_images):
            super().__init__()
            self.encoder = encoder
            self.n_images = n_images

        def forward(self, x):
            """
            Reimplementation of MnistNN forward method to allow pyvene to interact with the desired concept layer.
            Args:
                x: image tensor of size [bs, 1, 28, 56]

            Returns: This module outputs the hidden concept layer [bs, 1, 20] that will be rotated by DAS.
            """
            cs = []
            xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)  # splits images into "n_images" digits
            for i in range(self.n_images):
                # xs[i] is always [bs, 1, 28, 56]
                lc, _, _ = self.encoder(xs[i])  # encodes each sub image xs into its latent representation lc [bs, 2/n_images, 10]
                cs.append(lc)
            # cs is a list of len n_images containing tensor([bs, 2/n_images, 10])
            concept_layer = torch.cat(cs, dim=1).view(x.size(0), 1, -1)  # Ensure final shape [bs, 1, 20] as expected by pyvene
            return concept_layer

    def __init__(self, wrapped_model: MnistNN):
        super().__init__()
        self.wrapped_model = wrapped_model
        self.device = wrapped_model.device
        self.h_dim = int(2 * self.wrapped_model.n_facts)  # We define it as 10 * 2 = 20
        assert self.h_dim % 2 == 0

        # Pyvene expects self.config
        self.config = SimpleNamespace(
            num_classes=19,
            h_dim=self.h_dim,  # Dimension of the concept layer
            problem_type="classification",
        )

        # Define the layer we want pyvene to intervene on
        self.concept_layer = self.ConceptExtractionBlock(self.wrapped_model.encoder, self.wrapped_model.n_images)

        # Define the module and dimensionality for intervention mapping with pyvene global vars
        pyvene.type_to_module_mapping[type(self)] = {
            "block_output": ("concept_layer", pyvene.models.constants.CONST_OUTPUT_HOOK),
        }  # pyvene expects either a nn.Modul (e.g. "concept_layer") or a nn.ModuleList (e.g. "h[%s]")
        pyvene.type_to_dimension_mapping[type(self)] = {
            "block_output": ("h_dim",),
        }  # pyvene expects the dimensionality of the module to be: [batch_size, 1, h_dim]

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        # Choose the input: if input_ids is provided, use it; otherwise, try inputs_embeds.
        if input_ids is not None:
            x = input_ids
        elif inputs_embeds is not None:
            x = inputs_embeds
        else:
            raise ValueError("No input provided. Please supply either input_ids or inputs_embeds.")
        concept_layer = self.concept_layer(x)  # returns the potentially intervened concepts

        # Classification head
        py = self.wrapped_model.classifier(concept_layer)  # output probabilities for the sums
        # Add a small offset against numerical instabilities
        py = py + 1e-5
        with torch.no_grad():
            Z = torch.sum(py, dim=-1, keepdim=True)
        py = py / Z

        logits = py
        return (logits,)

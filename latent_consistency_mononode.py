
import torch
import time
from typing import Literal, Optional, Union
from diffusers import LatentConsistencyModelPipeline, LCMScheduler
from invokeai.app.invocations.primitives import (
    ImageField,
    ImageOutput,
    ImageCollectionOutput,
    LatentsField,
    LatentsOutput,
    BoardField,
)
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    OutputField,
    UIType,
    WithMetadata,
    WithWorkflow,
    invocation,
    UIComponent,
)

from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin

TORCH_PRECISION = Literal["fp16", "fp32"]

@invocation("latent_consistency_mononode", title="Latent Consistency MonoNode", tags=["latent_consistency"], category="latent_consistency", version="1.0.0")
class LatentConsistencyInvocation(BaseInvocation,
                                  # WithMetadata,
                                  # WithWorkflow
                                  ):

    """Wrapper node around diffusers LatentConsistencyTxt2ImgPipeline"""
    prompt: str = InputField(description="The prompt to use", ui_component=UIComponent.Textarea)
    num_inference_steps: int = InputField(description="The number of inference steps to use, 4-8 recommended", default=8)
    guidance_scale: float = InputField(description="The guidance scale to use", default=8.0)
    batches: int = InputField(description="The number of batches to use", default=1)
    images_per_batch: int = InputField(description="The number of images per batch to use", default=1)
    seeds: list[int] = InputField(description="List of noise seeds to use", default=None)
    lcm_origin_steps: int = InputField(description="The lcm origin steps to use", default=50)
    width: int = InputField(description="The width to use", default=512)
    height: int = InputField(description="The height to use", default=512)
    precision: TORCH_PRECISION = InputField(default="fp16", description="floating point precision")
    board: BoardField = InputField(default=None, description=FieldDescriptions.board, input=Input.Direct)

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
        dtype = torch.float16 if self.precision == "fp16" else torch.float32
        # trick of using LCMScheduler.from_pretrained() is from
        # https://huggingface.co/spaces/SimianLuo/Latent_Consistency_Model/blob/main/easy_run.py
        lcm_scheduler = LCMScheduler.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="scheduler")
        pipe = LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",
                                                              scheduler=lcm_scheduler,
                                                              torch_dtype=dtype,
                                                              )
        # From LatentConsistencyModelPipleine:
        #      To save GPU memory, torch.float16 can be used, but it may compromise image quality.
        pipe.to(torch_device="cuda", torch_dtype=dtype)
        all_images = []
        total_images = self.batches * self.images_per_batch
        start_inference = time.time()

        for i in range(self.batches):
            if self.seeds is not None and len(self.seeds) >= total_images:
                batch_seeds = self.seeds[i * self.images_per_batch : (i + 1) * self.images_per_batch]
                batch_generators = [torch.Generator(device="cuda").manual_seed(k) for k in batch_seeds]
            else:
                # batch_generators = [torch.Generator(device="cuda").manual_seed(k) for k in range(self.images_per_batch)]
                batch_generators = None   # with generators, pipeline will use default random seeds
            images = pipe(num_images_per_prompt=self.images_per_batch,
                          prompt=self.prompt,
                          generator=batch_generators,
                          num_inference_steps=self.num_inference_steps,
                          width=self.width,
                          height=self.height,
                          guidance_scale=self.guidance_scale,
                          original_inference_steps=self.lcm_origin_steps,
                          output_type="pil").images
            all_images = all_images + images
        end_inference = time.time()
        inference_time = end_inference - start_inference
        context.services.logger.info(f"LCM inference time: {inference_time} seconds")
        context.services.logger.info(f"LCM inference per image: {inference_time / total_images} seconds")
        context.services.logger.info(f"LCM inference images/second: {total_images / inference_time} seconds")

        image_fields = []
        for image in all_images:
            image_dto = context.services.images.create(
                image=image,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
                board_id=self.board.board_id if self.board else None,
                node_id=self.id,
                session_id=context.graph_execution_state_id,
                is_intermediate=False,
                metadata=None,
                workflow=None,
            )
            image_fields.append(ImageField(image_name=image_dto.image_name))

        return ImageCollectionOutput(collection=image_fields)

# # SaveImageCollectionInvocation not yet working
# @invocation(
#     "save_image_collection",
#     title="Save Image Collection",
#     tags=["image"],
#     category="image",
#     version="1.0.0",
#     use_cache=False,
# )
# class SaveImageCollectionInvocation(BaseInvocation, WithWorkflow, WithMetadata):
#     """Saves an image. Unlike an image primitive, this invocation stores a copy of the image."""
#
#     images: list[ImageField] = InputField(description=FieldDescriptions.image)
#     board: BoardField = InputField(default=None, description=FieldDescriptions.board, input=Input.Direct)
#
#     def invoke(self, context: InvocationContext) -> ImageOutput:
#         print("SaveImageCollection invoke(), num images: ", len(self.images))
#         new_image_fields = []
#         for image_field in self.images:
#             print("ImageField:")
#             print(image_field)
#             image = context.services.images.get_pil_image(image_field.image_name)
#
#             image_dto = context.services.images.create(
#                 image=image,
#                 image_origin=ResourceOrigin.INTERNAL,
#                 image_category=ImageCategory.GENERAL,
#                 board_id=self.board.board_id if self.board else None,
#                 node_id=self.id,
#                 session_id=context.graph_execution_state_id,
#                 is_intermediate=self.is_intermediate,
#                 metadata=None,
#                 workflow=None,
#             )
#             new_image_fields.append(ImageField(image_name=image_dto.image_name))
#         print(len(new_image_fields))
#         print(new_image_fields)
#
#         return ImageCollectionOutput(
#             collection=new_image_fields,
#         )

import os
import cv2
import torch
import logging
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from torch.utils.data import Dataset
import multiprocessing


class DeepfakeDataset(Dataset):
    def __init__(
        self,
        metadata_df,
        transform=None,
        frames_per_video=20,
        image_size=224,
        log_interval=100,
    ):
        self.metadata = metadata_df
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.image_size = image_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Initialize MTCNN here. Note: This can be slow.
        # If performance is an issue, consider passing a pre-initialized MTCNN object.
        self.mtcnn = MTCNN(
            image_size=self.image_size,
            margin=20,
            post_process=False,
            device=self.device,
            select_largest=True  # Handle multiple faces per frame,
            selection_method="probability"  # Select the face with the highest confidence
        )

        # Logging setup for multiprocessing
        self.log_interval = log_interval
        self.counter = multiprocessing.Value("i", 0)
        self.lock = multiprocessing.Lock()

    def __len__(self):
        return len(self.metadata)

    def extract_faces_from_video(self, video_path):
        if not os.path.exists(video_path):
            logging.warning(f"Video file not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            logging.warning(f"Video has zero frames: {video_path}. Skipping.")
            cap.release()
            return None

        if frame_count > self.frames_per_video:
            frame_indices = np.linspace(
                0, frame_count - 1, self.frames_per_video, dtype=int
            )
        else:
            frame_indices = np.arange(frame_count)

        frames_to_process = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames_to_process.append(pil_image)

        cap.release()

        if not frames_to_process:
            logging.warning(f"Could not read any frames from {video_path}. Skipping.")
            return None
        
        faces_list = []
        try:
            # Process each frame individually for maximum stability
            for frame in frames_to_process:
                detected_faces = self.mtcnn(frame)
                faces_list.append(detected_faces)
        except Exception as e:
            logging.error(f"Face extraction failed for a frame in {video_path} with error: {e}")
            return None
        
        # The rest of the logic to process 'faces_list' (renamed from faces_tensor)
        # should handle the list of tensors/None values correctly, as we developed.
        if not any(face is not None for face in faces_list):
            return None
        
        processed_faces = []
        for face in faces_list:
            if face is not None:
                if isinstance(face, list) and len(face) > 0:
                    processed_faces.append(face[0])
                elif not isinstance(face, list):
                    processed_faces.append(face)
        
        if not processed_faces:
            return None
        
        faces_tensor = torch.stack(processed_faces)
        return faces_tensor

    def __getitem__(self, idx):
        with self.lock:
            self.counter.value += 1
            count = self.counter.value
            if count % self.log_interval == 0:
                logging.info(f"Processing video {count}/{self.__len__()}...")

        row = self.metadata.iloc[idx]
        video_path = row["filepath"]
        label = 1 if row["label"] == "fake" else 0

        faces = self.extract_faces_from_video(video_path)

        if faces is None:
            logging.warning(f"No faces were extracted for {video_path}. Skipping this sample.")
            return None

        num_detected_faces = faces.shape[0]
        if num_detected_faces < self.frames_per_video:
            shortfall = self.frames_per_video - num_detected_faces
            last_face = faces[-1].unsqueeze(0)
            padding = last_face.repeat(shortfall, 1, 1, 1)
            faces = torch.cat([faces, padding], dim=0)

        if faces.shape[0] > self.frames_per_video:
            faces = faces[:self.frames_per_video]

        # Apply transforms to each frame individually
        if self.transform:
            transformed_faces = torch.stack([self.transform(frame) for frame in faces])
        else:
            transformed_faces = faces

        if torch.isnan(faces).any() or torch.isinf(faces).any():
            logging.warning(f"NaN or Inf detected in face tensor for {video_path}. Skipping sample.")
            return None

        return faces, torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    """
    Custom collate function to filter out None samples from the batch.
    This is essential for handling videos where face extraction fails.
    """
    # Filter out samples that returned None
    batch = [b for b in batch if b is not None]
    if not batch:
        # If the entire batch failed, return empty tensors
        return torch.tensor([]), torch.tensor([])
    
    # Use the default collate function on the cleaned batch
    return torch.utils.data.dataloader.default_collate(batch)
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize MTCNN here. Note: This can be slow.
        # If performance is an issue, consider passing a pre-initialized MTCNN object.
        self.mtcnn = MTCNN(
            image_size=self.image_size,
            margin=20,
            device=self.device,
            min_face_size=20,  # Ignores tiny, unlikely face detections
            thresholds=[0.6, 0.7, 0.7],  # Standard confidence thresholds
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
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            return []

        # --- Frame selection logic (no changes here) ---
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            cap.release()
            return []

        if frame_count > self.frames_per_video:
            frame_indices = np.linspace(
                0, frame_count - 1, self.frames_per_video, dtype=int
            )
        else:
            frame_indices = np.arange(frame_count)

        # --- Frame extraction and conversion to PIL (no changes here) ---
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
            return []

        # --- NEW: Two-step face detection, cropping, and resizing ---
        extracted_face_images = []
        try:
            for frame_pil in frames_to_process:
                boxes, probs = self.mtcnn.detect(frame_pil)

                if boxes is not None and len(boxes) > 0:
                    best_face_index = np.argmax(probs)
                    best_face_prob = probs[best_face_index]

                    if best_face_prob > 0.90:  # Confidence threshold
                        box = boxes[best_face_index]
                        face = frame_pil.crop(box)
                        face_resized = face.resize((self.image_size, self.image_size))
                        extracted_face_images.append(face_resized)
                    else:
                        extracted_face_images.append(None)
                else:
                    extracted_face_images.append(None)
        except Exception as e:
            logging.error(f"Face extraction failed for {video_path} with error: {e}")
            return []

        # Filter out frames where no good face was found
        valid_face_images = [f for f in extracted_face_images if f is not None]
        return valid_face_images

    def __getitem__(self, idx):
        with self.lock:
            self.counter.value += 1
            count = self.counter.value
            if count % self.log_interval == 0:
                logging.info(f"Processing video {count}/{self.__len__()}...")

        row = self.metadata.iloc[idx]
        video_path = row["filepath"]
        label = 1 if row["label"] == "fake" else 0

        face_images = self.extract_faces_from_video(video_path)

        if not face_images:
            logging.warning(f"No valid faces found for {video_path}. Skipping sample.")
            return None  # collate_fn will handle this

        num_detected_faces = len(face_images)
        if num_detected_faces < self.frames_per_video:
            shortfall = self.frames_per_video - num_detected_faces
            last_face_image = face_images[-1]
            face_images.extend([last_face_image] * shortfall)

        if len(face_images) > self.frames_per_video:
            face_images = face_images[: self.frames_per_video]

        # --- Apply transforms and stack into a single tensor ---
        if self.transform:
            faces_tensor = torch.stack([self.transform(img) for img in face_images])
        else:
            # Fallback to just converting to tensor if no transform is provided
            from torchvision import transforms as T

            faces_tensor = torch.stack([T.ToTensor()(img) for img in face_images])

        if torch.isnan(faces_tensor).any() or torch.isinf(faces_tensor).any():
            logging.warning(
                f"NaN or Inf detected in tensor for {video_path}. Skipping."
            )
            return None

        return faces_tensor, torch.tensor(label, dtype=torch.float32)


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

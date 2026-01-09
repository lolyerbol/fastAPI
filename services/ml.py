import os
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from services.database import engine

logger = logging.getLogger(__name__)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Model storage directory
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

TF_MODEL_DIR = MODEL_DIR 

TF_MODEL_PATH = MODEL_DIR / "my_model.keras"
SCALER_PATH = MODEL_DIR / "rush_hour_scaler.pkl"

# Feature list used by the model
FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "average_speed_mph",
    "trip_distance",
    "fare_amount",
    'is_night',
    'is_airport_pickup'
]


class SurchargePredictionRequest(BaseModel):
    tpep_pickup_datetime: datetime
    trip_distance: float
    trip_duration_minutes: float
    fare_amount: float
    pickup_location_id: int
    


class RushHourTFModel:
    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.input_dim = len(FEATURE_NAMES)

    def load_training_data(self) -> pd.DataFrame:
        query = """
        SELECT
            tpep_pickup_datetime,
            trip_distance,
            trip_duration_minutes,
            pickup_location_id,
            fare_amount,
            extra
        FROM nyc_taxi_trips
        WHERE tpep_pickup_datetime IS NOT NULL
          AND trip_distance IS NOT NULL
          AND trip_duration_minutes IS NOT NULL
        LIMIT 1000
        """
        try:
            df = pd.read_sql(query, engine)
            logger.info(f"Loaded {len(df)} rows from database for training.")
            return df
        except Exception as e:
            logger.exception("Failed to load training data from DB")
            raise

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df["hour"] = df["tpep_pickup_datetime"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["is_night"] = ((df["hour"] >= 20) | (df["hour"] < 6)).astype(int)
        airport_zones = {132, 138}  # example: JFK, LGA
        df["is_airport_pickup"] = df["pickup_location_id"].isin(airport_zones).astype(int)
        df["day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        # safe average speed: if duration <= 0 then set speed 0
        df["average_speed_mph"] = np.where(
            df["trip_duration_minutes"] > 0,
            df["trip_distance"] / (df["trip_duration_minutes"] / 60.0),
            0.0,
        )
        df["has_extra_surcharge"] = (df.get("extra", 0) > 0).astype(int)
        return df

    def prepare_xy(self, df: pd.DataFrame):
        df = self.engineer_features(df)
        df = df.dropna(subset=FEATURE_NAMES + ["has_extra_surcharge"])
        X = df[FEATURE_NAMES].astype(float)
        y = df["has_extra_surcharge"].astype(int)
        return X, y

    def build_model(self, X_train) -> tf.keras.Model:
        # Very small, stable MLP to avoid training instability on small/imbalanced data
        tf.keras.backend.clear_session()
        #model = tf.keras.Sequential(
        #        tf.keras.layers.Input(shape=(self.input_dim,)),
        #    [
        #        tf.keras.layers.Dense(32, activation="relu"),
        #        tf.keras.layers.Dense(8, activation="relu"),
        #        tf.keras.layers.Dense(1, activation="sigmoid"),
        #    ]
        #)
        #model.compile(
        #    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        #    loss="binary_crossentropy",
        #    metrics=["accuracy"],
        #)
        model = tf.keras.Sequential([
                    tf.keras.layers.Dense(1, activation="sigmoid", input_shape=(X_train.shape[1],))
                    ])

        model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                    loss="binary_crossentropy",
                    metrics=[tf.keras.metrics.AUC(name="auc")]
                    )
        return model

    def train(self, epochs: int = 1) -> None:
        logger.info("Starting TensorFlow model training (simplified)...")
        df = self.load_training_data()
        X, y = self.prepare_xy(df)
#        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
#        logger.info(X.head())
#        logger.info(y.value_counts().to_dict())
        if len(X) == 0:
            raise ValueError("No training data available")

        # Basic cleaning: replace inf/nan and fill missing
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
        )

        # If the training target has only one class, create a trivial predictor
        if y_train.nunique() == 1:
            majority = int(y_train.iloc[0])
            logger.warning("Only one class in training labels; creating constant-output model.")
            # trivial keras model that outputs constant probability
            self.model = tf.keras.Sequential([tf.keras.layers.Input(shape=(self.input_dim,)),
                                              tf.keras.layers.Dense(1, activation="sigmoid")])
            # set weights so output ~ majority
            w = np.zeros((self.input_dim, 1))
            b = np.array([1.0 if majority == 1 else 0.0])
            self.model.build((None, self.input_dim))
            self.model.set_weights([w, b])
            # still save an (empty) scaler fitted to training features
            self.scaler = StandardScaler()
            self.scaler.fit(X_train)
            with open(SCALER_PATH, "wb") as f:
                pickle.dump(self.scaler, f)
            TF_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            self.model.save(str(TF_MODEL_DIR), overwrite=True)
            logger.info("Saved trivial constant model.")
            return

        # Fit scaler and persist
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)

        # Build compact model
        self.model = self.build_model(X_train)

        # Safeguards for unstable training
        callbacks = [
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ]

        # Compute simple class weight (avoid division by zero)
        positives = int(y_train.sum())
        negatives = int(len(y_train) - positives)
        class_weight = None
        if positives > 0:
            class_weight = {0: 1.0, 1: max(1.0, negatives / max(1.0, positives))}

        try:
            self.model.fit(X_train_scaled,
                y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=epochs,
                class_weight=class_weight,
                verbose=1,)

        except Exception as e:
            logger.exception("Training failed; attempting fallback to very small training run.")

        # Save model
        TF_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.model.save(str(TF_MODEL_PATH), overwrite=True)
        logger.info(f"Saved TensorFlow model to {TF_MODEL_PATH}")

    def save_loaded_scaler(self):
        if self.scaler is not None:
            with open(SCALER_PATH, "wb") as f:
                pickle.dump(self.scaler, f)

    def load(self) -> bool:
    # Load scaler
        if SCALER_PATH.exists():
            try:
                with open(SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)
            except Exception:
                logger.exception("Failed to load scaler")
                self.scaler = None
        else:
            self.scaler = None

    # Load TensorFlow model (.keras file)
        if TF_MODEL_PATH.exists():
            try:
                self.model = tf.keras.models.load_model(TF_MODEL_PATH)
                logger.info("TensorFlow model loaded from disk.")
                return True
            except Exception:
                logger.exception("Failed to load TensorFlow model")
                self.model = None
                return False

        logger.info("No TensorFlow model found on disk.")
        return False


    def predict_single(self, request: SurchargePredictionRequest) -> Dict[str, Any]:
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not loaded. Call load or train first.")
        logger.info("Preparing features for prediction...")
        df = pd.DataFrame(
            [
                {
                    "tpep_pickup_datetime": request.tpep_pickup_datetime,
                    "trip_distance": request.trip_distance,
                    "trip_duration_minutes": request.trip_duration_minutes,
                    "fare_amount": request.fare_amount,
                    "extra": 0,
                    "pickup_location_id": request.pickup_location_id,
                }
            ]
        )
        logger.info("Engineering features...")
        df = self.engineer_features(df)
        X = df[FEATURE_NAMES].copy()

        logger.info("Scaling features...")
        X_scaled = self.scaler.transform(X)
        logger.info("Running prediction...")
        X_tensor = tf.convert_to_tensor(X_scaled)

# 2. Call the model directly (The __call__ method)
# 'training=False' ensures dropout/batch-norm layers are in inference mode
        predictions = self.model(X_tensor, training=False)

# 3. Extract the single value
# .numpy() converts the tensor back to a format you can use easily
        proba = float(predictions.numpy()[0][0])
        #proba = self.model.predict(X_scaled)
        pred = int(proba >= 0.5)
        logger.info(f"Prediction: {pred} with probability {proba:.4f}")
        return {
            "has_extra_surcharge": pred,
            "probability_extra_surcharge": proba,
            "probability_no_extra_surcharge": 1.0 - proba,
            "hour": int(df["hour"].iloc[0]),
            "day_of_week": int(df["day_of_week"].iloc[0]),
            "is_weekend": int(df["is_weekend"].iloc[0]),
            "average_speed_mph": float(df["average_speed_mph"].iloc[0]),
            "trip_distance": float(df["trip_distance"].iloc[0]),
            "fare_amount": float(df["fare_amount"].iloc[0]),
            "is_night": int(df["is_night"].iloc[0]),
            "is_airport_pickup": int(df["is_airport_pickup"].iloc[0]),
        }


# Global model instance
_rush_hour_model: Optional[RushHourTFModel] = None  # type: ignore


def load_surcharge_model() -> None:
    """
    Initialize the global TF model instance. If model files are missing, training will be triggered.
    This function is safe to call on startup.
    """
    global _rush_hour_model
    if _rush_hour_model is None:
        _rush_hour_model = RushHourTFModel()

    if not _rush_hour_model.load():
        logger.info("No existing TF model found — starting training(?).")
        #_rush_hour_model.train()
        pass


def predict_surcharge(request: SurchargePredictionRequest) -> Dict[str, Any]:
 #   print("DEBUG: In predict_surcharge function")
    if _rush_hour_model is None:
        raise RuntimeError("Model not initialized. Call load_surcharge_model() first.")
    return _rush_hour_model.predict_single(request)
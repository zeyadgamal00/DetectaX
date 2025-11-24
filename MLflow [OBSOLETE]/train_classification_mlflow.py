# MLflow/train_classification_mlflow_fixed.py
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# Add the MLflow directory to the path so we can import our modules
sys.path.append('../MLflow')

import mlflow
from tracking_setup import setup_mlflow_tracking, start_experiment_run
from experiment_logging import log_training_metrics, log_classification_artifacts
from model_registry import ModelRegistry

def train_classification_with_mlflow():
    """
    Enhanced training script for classification with MLflow tracking using nested runs
    """
    print("🚀 Starting Classification Training with MLflow (Nested Runs)...")
    
    # Setup MLflow
    ws = setup_mlflow_tracking()
    
    # Start main parent run
    with start_experiment_run("cifar10-classification", "mobileNetV2_complete"):
        
        # Load and prepare data
        print("📥 Loading CIFAR-10 dataset...")
        (train_data, val_data, test_data), data_info = tfds.load("cifar10", 
                                            split=['train[10000:]', 'train[0:10000]', 'test'],
                                            as_supervised=True, with_info=True, shuffle_files=True)
        
        classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        
        # Data augmentation and preprocessing
        data_augmentation = keras.Sequential([
            layers.Resizing(224, 224),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomBrightness(0.1),
            layers.RandomContrast(0.1),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.Rescaling(1.0 / 255.0) 
        ])
        
        def augment(image, label):
            return data_augmentation(image), label
        
        def preprocess(image, label):
            image = tf.image.resize(image, [224, 224])
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        
        train_data = train_data.map(augment, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
        val_data = val_data.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)
        test_data = test_data.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)

        # Log global training parameters in parent run
        training_params = {
            "batch_size": 32,
            "base_model": "MobileNetV2",
            "dataset": "CIFAR-10",
            "fine_tune_layers": 50,
            "early_stopping_patience": 3
        }
        mlflow.log_params(training_params)
        
        # PHASE 1: Transfer Learning with Frozen Base Model (Nested Run)
        print("Phase 1: Transfer Learning (Frozen Base)")
        with mlflow.start_run(run_name="phase1_transfer_learning", nested=True):
            
            # Log Phase 1 specific parameters
            phase1_params = {
                "phase": "transfer_learning",
                "learning_rate": 0.001,
                "trainable_layers": 0,
                "epochs": 5,
                "optimizer": "adam"
            }
            mlflow.log_params(phase1_params)
            
            # Use a pretrained base model
            base_model = keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            
            # Add custom classifier on top
            model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.2),
                layers.Dense(10, activation='softmax')
            ])

            # Compile the model
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
                keras.callbacks.ModelCheckpoint('best_model_phase1.h5', monitor='val_loss', save_best_only=True)
            ]

            # Train Phase 1
            history_phase1 = model.fit(
                train_data,
                epochs=phase1_params["epochs"],
                validation_data=val_data,
                callbacks=callbacks,
                verbose=1
            )

            # Log Phase 1 metrics
            final_metrics_phase1 = {
                "train_accuracy": history_phase1.history['accuracy'][-1],
                "val_accuracy": history_phase1.history['val_accuracy'][-1],
                "train_loss": history_phase1.history['loss'][-1],
                "val_loss": history_phase1.history['val_loss'][-1]
            }
            log_training_metrics(final_metrics_phase1)
            
            # Log training history for Phase 1
            log_classification_artifacts(history_phase1, classes, "phase1_artifacts")
            
            # Save Phase 1 model
            phase1_model_path = 'cifar10_model_phase1.keras'
            model.save(phase1_model_path)
            mlflow.log_artifact(phase1_model_path)
            
            print("Phase 1 (Transfer Learning) completed!")

        # PHASE 2: Fine-tuning (Nested Run)
        print("Phase 2: Fine-tuning (Unfrozen Layers)")
        with mlflow.start_run(run_name="phase2_fine_tuning", nested=True):
            
            # Log Phase 2 specific parameters
            phase2_params = {
                "phase": "fine_tuning",
                "learning_rate": 0.01,
                "trainable_layers": 50,
                "epochs": 5,
                "optimizer": "adam"
            }
            mlflow.log_params(phase2_params)
            
            # Unfreezing the last 50 layers of the model
            base_model.trainable = True
            fine_tune = 50
            
            # Freeze all the layers before the `fine_tune` layer
            for layer in base_model.layers[:-fine_tune]:
                layer.trainable = False

            # Recompile with lower learning rate for fine-tuning
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.01),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=["accuracy"]
            )
            
            # Train Phase 2
            history_phase2 = model.fit(
                train_data,
                epochs=phase2_params["epochs"],
                validation_data=val_data,
                callbacks=callbacks,
                verbose=1
            )

            # Log Phase 2 metrics
            final_metrics_phase2 = {
                "train_accuracy": history_phase2.history['accuracy'][-1],
                "val_accuracy": history_phase2.history['val_accuracy'][-1],
                "train_loss": history_phase2.history['loss'][-1],
                "val_loss": history_phase2.history['val_loss'][-1]
            }
            log_training_metrics(final_metrics_phase2)
            
            # Log training history for Phase 2
            log_classification_artifacts(history_phase2, classes, "phase2_artifacts")
            
            print("Phase 2 (Fine-tuning) completed!")

        # FINAL EVALUATION (Back in parent run)
        print("Evaluating final model...")
        
        # Save the final model
        final_model_path = 'cifar10_model_final_mlflow.keras'
        model.save(final_model_path)
        mlflow.log_artifact(final_model_path)

        # Evaluate the model
        test_loss, test_acc = model.evaluate(test_data)
        
        # Get comprehensive metrics
        y_true = []
        y_pred = []
        for images, labels in test_data:
            preds = model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(preds, axis=1))
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        final_metrics = {
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "precision_macro": precision_score(y_true, y_pred, average='macro'),
            "recall_macro": recall_score(y_true, y_pred, average='macro'),
            "f1_score_macro": f1_score(y_true, y_pred, average='macro')
        }
        
        # Log final metrics in parent run
        log_training_metrics(final_metrics)
        
        # Create and log confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - Final Model')
        plt.tight_layout()
        plt.savefig('confusion_matrix_final.png')
        mlflow.log_artifact('confusion_matrix_final.png')
        plt.close()
        
        # Log sample predictions
        sample_images, sample_labels = next(iter(test_data))
        sample_preds = model.predict(sample_images, verbose=0)
        sample_pred_labels = np.argmax(sample_preds, axis=1)
        
        plt.figure(figsize=(12, 6))
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            img = sample_images[i].numpy() * 255.0
            img = np.clip(img, 0, 255).astype("uint8")
            plt.imshow(img)
            true_cls = classes[sample_labels[i]]
            pred_cls = classes[sample_pred_labels[i]]
            color = 'green' if true_cls == pred_cls else 'red'
            plt.title(f"True: {true_cls}\nPred: {pred_cls}", color=color, fontsize=8)
            plt.axis("off")
        plt.tight_layout()
        plt.savefig('sample_predictions_final.png')
        mlflow.log_artifact('sample_predictions_final.png')
        plt.close()

        # Register the final model
        if ws:
            registry = ModelRegistry(ws)
            registry.register_classification_model(
                final_model_path,
                mlflow.active_run().info.run_id,
                final_metrics,
                "CIFAR-10 Classification Model - MobileNetV2 with Fine-tuning (Nested Runs)"
            )
        
        print("Classification training completed with MLflow tracking!")
        print(f"Final Test Accuracy: {test_acc:.4f}")
        
        return model, final_metrics

if __name__ == "__main__":
    model, metrics = train_classification_with_mlflow()
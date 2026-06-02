import tensorflow as tf
import numpy as np

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

from vit_keras import vit

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# =========================================================
# DATA GENERATORS
# =========================================================

def get_data_generators(base_dir):

    train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1
).flow_from_directory(
        f"{base_dir}/train",
        target_size=(224, 224),
        batch_size=8,
        class_mode='binary'
    )

    val_gen = ImageDataGenerator(
        rescale=1./255
    ).flow_from_directory(
        f"{base_dir}/valid",
        target_size=(224, 224),
        batch_size=8,
        class_mode='binary',
        shuffle=False
    )

    test_gen = ImageDataGenerator(
        rescale=1./255
    ).flow_from_directory(
        f"{base_dir}/test",
        target_size=(224, 224),
        batch_size=8,
        class_mode='binary',
        shuffle=False
    )

    return train_gen, val_gen, test_gen


# =========================================================
# BUILD VIT MODEL
# =========================================================

def build_vit():

    vit_model = vit.vit_b16(
        image_size=224,
        pretrained=True,
        include_top=False,
        pretrained_top=False
    )

    for layer in vit_model.layers:
        layer.trainable = False

    x = tf.keras.layers.Flatten()(vit_model.output)

    x = tf.keras.layers.Dense(
        256,
        activation='relu'
    )(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    output = tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        dtype='float32'
    )(x)

    model = tf.keras.Model(
        inputs=vit_model.input,
        outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model, vit_model


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(generator, model, name="Test"):

    print(f"\n📊 {name} Results")

    preds = model.predict(generator)

    preds_binary = (preds > 0.5).astype(int)

    y_true = generator.classes

    accuracy = accuracy_score(
        y_true,
        preds_binary
    )

    precision = precision_score(
        y_true,
        preds_binary
    )

    recall = recall_score(
        y_true,
        preds_binary
    )

    f1 = f1_score(
        y_true,
        preds_binary
    )

    auc = roc_auc_score(
        y_true,
        preds
    )

    print(f"\n✅ Accuracy  : {accuracy:.4f}")
    print(f"✅ Precision : {precision:.4f}")
    print(f"✅ Recall    : {recall:.4f}")
    print(f"✅ F1-Score  : {f1:.4f}")
    print(f"✅ ROC-AUC   : {auc:.4f}")

    print("\n📊 Classification Report\n")

    print(
        classification_report(
            y_true,
            preds_binary
        )
    )

    print("\n📊 Confusion Matrix\n")

    print(
        confusion_matrix(
            y_true,
            preds_binary
        )
    )

    return accuracy, precision, recall, f1, auc


# =========================================================
# TRAINING PIPELINE
# =========================================================

def train():

    BASE_DIR = r"C:\Users\imabb\Downloads\gitdemo\passion-project-deep-fake-identifiers\deepfake_dataset\real-vs-fake"

    print("\n🚀 Vision Transformer Training Pipeline")

    train_gen, val_gen, test_gen = get_data_generators(BASE_DIR)

    model, vit_model = build_vit()

    callbacks = [

        EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=1,
            verbose=1
        ),

        ModelCheckpoint(
            filepath=r"C:\Users\imabb\Downloads\gitdemo\passion-project-deep-fake-identifiers\src\models\vit\best_vit_model.keras",
            save_best_only=True,
            monitor='val_loss'
        )
    ]

    # =====================================================
    # PHASE 1
    # =====================================================

    print("\n🔥 PHASE 1: Transfer Learning")

    train_steps = len(train_gen) // 6
    val_steps = len(val_gen) // 6

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=2,
        callbacks=callbacks
    )

    train_acc_phase1 = history1.history['accuracy'][-1]

    print(f"\n✅ Training Accuracy (Phase 1): {train_acc_phase1:.4f}")

    # =====================================================
    # PHASE 2
    # =====================================================

    print("\n🔥 PHASE 2: Fine-Tuning")

    for layer in vit_model.layers[-40:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=2,
        callbacks=callbacks
    )

    final_train_acc = history2.history['accuracy'][-1]

    print(f"\n🔥 Final Training Accuracy: {final_train_acc:.4f}")

    # =====================================================
    # VALIDATION
    # =====================================================

    val_acc, val_prec, val_rec, val_f1, val_auc = evaluate_model(
        val_gen,
        model,
        "Validation"
    )

    # =====================================================
    # TEST
    # =====================================================

    test_acc, test_prec, test_rec, test_f1, test_auc = evaluate_model(
        test_gen,
        model,
        "Test"
    )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n📊 FINAL SUMMARY")

    print(f"\n🔥 Training Accuracy : {final_train_acc:.4f}")

    print(f"\n✅ Validation Accuracy : {val_acc:.4f}")
    print(f"✅ Validation ROC-AUC  : {val_auc:.4f}")

    print(f"\n✅ Test Accuracy       : {test_acc:.4f}")
    print(f"✅ Test Precision      : {test_prec:.4f}")
    print(f"✅ Test Recall         : {test_rec:.4f}")
    print(f"✅ Test F1-Score       : {test_f1:.4f}")
    print(f"✅ Test ROC-AUC        : {test_auc:.4f}")

    # =====================================================
    # SAVE MODEL
    # =====================================================

    save_path = r"C:\Users\imabb\Downloads\gitdemo\passion-project-deep-fake-identifiers\src\models\vit\vit_b16_finetuned.keras"

    model.save(save_path)

    print(f"\n💾 Model Saved: {save_path}")


# =========================================================

if __name__ == "__main__":
    train()
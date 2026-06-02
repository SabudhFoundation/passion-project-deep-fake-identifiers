from sklearn.neural_network import MLPClassifier

def build_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),  # 🔥 deeper network
        activation='relu',
        solver='adam',
        batch_size=256,                      
        max_iter=200,                       
        learning_rate='adaptive',
        early_stopping=True,
        verbose=True,
        random_state=42
    )
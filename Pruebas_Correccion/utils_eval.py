from sklearn.metrics import classification_report
import numpy as np

def evaluar_modelo(nombre, modelo, X, y, K=5, es_mlp=False, epochs=15, batch_size=64, lr=1e-3, random_state=42):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
    resultados = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ------------------ ENTRENAMIENTO ------------------
        if es_mlp:
            input_dim, num_classes = X.shape[1], len(np.unique(y))
            class MLP(nn.Module):
                def __init__(self, input_dim, num_classes):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, num_classes)
                    )
                def forward(self, xb): return self.net(xb)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MLP(input_dim, num_classes).to(device)
            crit = nn.CrossEntropyLoss()
            opt  = torch.optim.Adam(model.parameters(), lr=lr)

            train_dl = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                torch.tensor(y_train, dtype=torch.long)), batch_size=batch_size, shuffle=True)
            test_dl  = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                               torch.tensor(y_test, dtype=torch.long)), batch_size=batch_size)

            for _ in range(epochs):
                model.train()
                for xb, yb in train_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad(); loss = crit(model(xb), yb)
                    loss.backward(); opt.step()

            model.eval()
            y_pred = []
            with torch.no_grad():
                for xb, _ in test_dl:
                    xb = xb.to(device)
                    y_pred += model(xb).argmax(1).cpu().tolist()
            y_true = y_test.tolist()

        else:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            y_true = y_test

        # ------------------ MÉTRICAS ------------------
        oa  = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")
        cm  = confusion_matrix(y_true, y_pred, labels=np.unique(y))

        # F1 por clase
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        f1_clases = {int(k): v["f1-score"] for k, v in report.items() if k.isdigit()}
        clase_dificil = min(f1_clases, key=f1_clases.get)  # clase con peor F1

        print(f"[{nombre}] Fold {fold}: OA={oa:.3f}, F1m={f1m:.3f}, Clase más difícil={clase_dificil}")

        resultados.append({
            "fold": fold,
            "OA": oa,
            "F1_macro": f1m,
            "F1_clases": f1_clases,
            "clase_dificil": clase_dificil,
            "matriz_confusion": cm
        })

    return resultados
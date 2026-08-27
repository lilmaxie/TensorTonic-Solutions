import numpy as np

def tanh(x):
    return np.tanh(x)

class BertPooler:
    """
    BERT Pooler: Extracts [CLS] and applies dense + tanh.
    """
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.W = np.random.randn(hidden_size, hidden_size) * 0.02
        self.b = np.zeros(hidden_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Returns: np.ndarray of shape (batch, hidden_size) with tanh-activated [CLS] output
        """
        hs = np.asarray(hidden_states, dtype=np.float64)

        # extract the [CLS] vector at index 0 along the sequence axis (seq_len) --> shape (batch, hidden_size)
        cls_hidden = hs[:, 0, :]

        W_mat = np.asarray(self.W, dtype=np.float64)
        b_vec = np.asarray(self.b, dtype=np.float64)

        # linear transformation: cls_hidden @ W + b
        linear_out = cls_hidden@W_mat + b_vec
        pooled_output = tanh(linear_out)

        return pooled_output

class SequenceClassifier:
    """
    Sequence classification head on top of BERT.
    """
    
    def __init__(self, hidden_size: int, num_classes: int):
        self.pooler = BertPooler(hidden_size)
        self.classifier = np.random.randn(hidden_size, num_classes) * 0.02
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Returns: np.ndarray of shape (batch, num_classes) with classification logits
        """
        pooled_output = self.pooler.forward(hidden_states)

        # pass through the classifier matrix to obtain classification logits --> shape: (batch, num_classes)
        W_clf = np.asarray(self.classifier, dtype=np.float64)
        logits = pooled_output @ W_clf

        return logits
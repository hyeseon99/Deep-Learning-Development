import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

# 사용자 지정 파라미터 설정
max_len = 64  # 텍스트가 길다면 max_len을 늘립니다
batch_size = 32  # 배치 크기 조정
warmup_ratio = 0.1  # warmup 비율 조정
num_epochs = 100  # 더 많은 학습 에포크
max_grad_norm = 1.0
learning_rate = 1e-5  # 더 높은 학습률
model_save_path = "/content/drive/MyDrive/kobert2_model.pt"


# 사용자 지정 파라미터 설정
max_len = 64
batch_size = 32
warmup_ratio = 0.1
num_epochs = 100
max_grad_norm = 1.0
learning_rate = 1e-5
model_save_path = "/content/drive/MyDrive/kobert2_model.pt"

# Early Stopping 설정
early_stopping_patience = 3
best_val_loss = float('inf')
early_stopping_counter = 0

# Tokenizer 및 모델 초기화 (KOBERT)
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")  # AutoTokenizer 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset 클래스 정의
class MyDataset(Dataset):
    def __init__(self, data, labels, tokenizer, max_len):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = str(self.data[idx])
        label = self.labels[idx]

        # 토큰화 및 패딩, 잘라내기 적용
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        token_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return token_ids, attention_mask, label

# 정확도 계산 함수
def calc_accuracy(logits, labels):
    preds = torch.argmax(logits, axis=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

# F1 스코어 계산 함수
def calc_f1_score(logits, labels):
    preds = torch.argmax(logits, axis=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return f1_score(labels, preds, average='weighted')

# 데이터 준비 및 가중치 계산
data_list = []
for q, label in zip(chatbot_data['사람문장1'], chatbot_data['감정_대분류']):
    data_list.append([q, int(label)])

texts = [item[0] for item in data_list]
labels = [item[1] for item in data_list]

# 데이터셋 분할 (8:2 비율)
texts_train, texts_val, labels_train, labels_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 레이블 클래스 수 확인 및 클래스 가중치 계산
num_labels = len(set(labels))
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# 데이터셋 및 데이터로더 생성
train_dataset = MyDataset(texts_train, labels_train, tokenizer, max_len)
val_dataset = MyDataset(texts_val, labels_val, tokenizer, max_len)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# KOBERT 모델 구성
config = BertConfig.from_pretrained("monologg/kobert", num_labels=num_labels, hidden_dropout_prob=0.3)  # dropout 비율 조정
model = BertForSequenceClassification.from_pretrained("monologg/kobert", config=config)
model.to(device)

# 옵티마이저 및 스케줄러 설정
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
warmup_steps = int(total_steps * warmup_ratio)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# 손실 함수 정의
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_f1 = 0.0
    for batch_id, (token_ids, attention_mask, label) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)

        # 모델에 입력하여 출력 얻기
        outputs = model(input_ids=token_ids, attention_mask=attention_mask, labels=label)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        train_acc += calc_accuracy(logits, label)
        train_f1 += calc_f1_score(logits, label)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss / len(train_dataloader):.4f}, Accuracy: {train_acc / len(train_dataloader):.4f}, F1 Score: {train_f1 / len(train_dataloader):.4f}")

    # 검증 단계
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_f1 = 0.0
    with torch.no_grad():
        for token_ids, attention_mask, label in val_dataloader:
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)

            outputs = model(input_ids=token_ids, attention_mask=attention_mask, labels=label)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()
            val_acc += calc_accuracy(logits, label)
            val_f1 += calc_f1_score(logits, label)

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc / len(val_dataloader):.4f}, Validation F1 Score: {val_f1 / len(val_dataloader):.4f}")

    # Early Stopping 검증
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
        # 최적 모델 저장
        torch.save(model.state_dict(), model_save_path)
        print(f"New best model saved to {model_save_path}")
    else:
        early_stopping_counter += 1
        print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered.")
        break

print("Training complete!")

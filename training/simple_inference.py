from transformers import pipeline

classifier = pipeline("text-classification", model="./models/final_ai_detector")

texts = [
    "Kubernetes использует декларативные конфигурации, что позволяет разработчикам определять желаемое состояние системы, а платформа автоматически управляет достижением этого состояния. Это делает Kubernetes мощным инструментом для управления сложными микросервисными архитектурами.",
    "Это мой текст, я не писал его с помощью AI!!",
]

results = classifier(texts)

for text, res in zip(texts, results):
    print(
        f"Текст: {text[:30]}... -> Метка: {res['label']}, Уверенность: {res['score']:.4f}"
    )

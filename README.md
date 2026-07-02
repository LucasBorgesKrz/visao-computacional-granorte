# 🚛 Visão Computacional — Granorte

Protótipo desenvolvido para a **Granorte** com o objetivo de automatizar
o monitoramento de caminhões na entrada da empresa usando visão computacional.

## 📋 O que o sistema faz

- 🔍 Detecta e lê automaticamente a **placa do caminhão** via OCR
- 🪣 Identifica se o **basculante está cheio** ao chegar na empresa
- 🔔 Emite **alerta sonoro automático** quando detecta basculante cheio
- 🎥 Gera vídeo processado com **bounding boxes** desenhados em tempo real

## 🛠️ Tecnologias utilizadas

| Tecnologia | Uso |
|---|---|
| YOLOv8 (Ultralytics) | Detecção de objetos |
| Roboflow | Anotação e gestão do dataset |
| EasyOCR | Leitura de texto das placas |
| OpenCV | Processamento de vídeo |
| Google Colab (GPU T4) | Treinamento do modelo |
| Python | Linguagem principal |

## 📁 Estrutura do projeto

├── notebooks/
│   └── deteccao_placas.ipynb       # Notebook principal (treino + detecção)
├── deteccao_basculante_cheio.py    # Detecção de basculante com alerta sonoro
├── detectar_carga.py               # Análise de carga por textura (heurística)
└── README.md

## 🚀 Como usar

1. Abra o arquivo `notebooks/deteccao_placas.ipynb` no **Google Colab**
2. Ative a GPU: `Ambiente de execução → Alterar tipo → T4 GPU`
3. Execute as células em ordem:
   - Célula 1: instala as dependências
   - Célula 2: cole o código do seu dataset no Roboflow
   - Célula 3: treina o modelo YOLOv8
   - Células seguintes: roda a detecção no vídeo
4. Para detecção com alerta sonoro use `deteccao_basculante_cheio.py`

## 📊 Resultados do protótipo

- **mAP50: 0.995** — precisão de detecção no conjunto de validação
- **mAP50-95: 0.856**
- Detecção funcionando em vídeo noturno de câmera de segurança

## 👤 Autor

**Lucas Borges** — Estagiário de Desenvolvimento na Granorte  
[LinkedIn](https://www.linkedin.com/in/lucas-borges-krziminski-b455593b3/) · [GitHub](https://github.com/LucasBorgesKrz)

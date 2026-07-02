import cv2
import numpy as np
from ultralytics import YOLO
from IPython.display import display, Audio
from google.colab.patches import cv2_imshow
import time

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════

MODEL_PATH = 'runs/detect/placas_caminhao/weights/best.pt'  # ajuste se necessário
model = YOLO(MODEL_PATH)

print('Classes do modelo:', model.names)
# Esperado algo como: {0: 'basculante cheio'}


# ══════════════════════════════════════════════════════════════════════════
# ALERTA SONORO
# ══════════════════════════════════════════════════════════════════════════

def alerta_sonoro(freq=1000, duracao=0.4, sr=22050):
    """Gera e toca um beep em memória (sem precisar de arquivo de áudio)."""
    t = np.linspace(0, duracao, int(sr * duracao), False)
    onda = np.sin(freq * t * 2 * np.pi)
    display(Audio(onda, rate=sr, autoplay=True))


# ══════════════════════════════════════════════════════════════════════════
# DETECÇÃO NO VÍDEO
# ══════════════════════════════════════════════════════════════════════════

def detectar_basculante_cheio(video_path, confianca=0.5, mostrar_frames=True,
                                intervalo_alerta=3, max_frames=None):
    """
    Detecta 'basculante cheio' no vídeo. Quando a detecção aparece:
      - desenha bounding box vermelho
      - escreve label "BASCULANTE CHEIO"
      - toca um beep de alerta (respeitando um intervalo mínimo entre beeps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('❌ Erro ao abrir vídeo:', video_path)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'📹 Vídeo: {total_frames} frames | {fps:.1f} FPS')

    deteccoes = []
    ultimo_alerta = -999  # garante que o primeiro alerta sempre toque
    frame_count = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('resultado_basculante.mp4', fourcc, fps,
                          (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_count >= max_frames:
            break

        frame_count += 1
        frame_anotado = frame.copy()
        tempo_atual = frame_count / fps

        # ── Detecção YOLO ──
        results = model(frame, conf=confianca, verbose=False)[0]

        detectou_cheio = False

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            detectou_cheio = True

            # ── Bounding box vermelho (alerta) ──
            cor = (0, 0, 255)
            cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), cor, 3)

            label = f'BASCULANTE CHEIO {conf:.0%}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame_anotado, (x1, y1 - th - 14), (x1 + tw + 10, y1), cor, -1)
            cv2.putText(frame_anotado, label, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            deteccoes.append({
                'frame': frame_count,
                'tempo_s': round(tempo_atual, 1),
                'confianca': conf
            })

        # ── Alerta sonoro (com intervalo mínimo entre beeps) ──
        if detectou_cheio and (tempo_atual - ultimo_alerta) > intervalo_alerta:
            print(f'🔔 [{tempo_atual:.1f}s] Caminhão chegou — BASCULANTE CHEIO detectado!')
            alerta_sonoro()
            ultimo_alerta = tempo_atual

        out.write(frame_anotado)

        if mostrar_frames and frame_count % 30 == 0:
            print(f'Frame {frame_count}/{total_frames}')
            cv2_imshow(frame_anotado)

    cap.release()
    out.release()

    print(f'\n✅ Processamento concluído!')
    print(f'📊 Total de frames com "basculante cheio": {len(deteccoes)}')
    if deteccoes:
        primeiro = deteccoes[0]
        print(f"\n🚛 Primeira detecção: {primeiro['tempo_s']}s (frame {primeiro['frame']}), confiança {primeiro['confianca']:.0%}")

    return deteccoes


# ══════════════════════════════════════════════════════════════════════════
# RODAR
# ══════════════════════════════════════════════════════════════════════════

deteccoes = detectar_basculante_cheio(
    video_path,           # variável já definida na célula de upload do vídeo
    confianca=0.5,
    mostrar_frames=True,
    intervalo_alerta=3    # segundos entre alertas repetidos
)

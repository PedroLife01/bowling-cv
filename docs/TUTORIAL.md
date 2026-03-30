# Bowling CV - Tutorial de Utilizacao

## Indice

1. [Instalacao](#instalacao)
2. [Gravacao de Videos](#gravacao-de-videos)
3. [Modo 1: Analise de Video Gravado (Terminal)](#modo-1-analise-de-video-gravado-terminal)
4. [Modo 2: Analise de Video Gravado (Interface Web)](#modo-2-analise-de-video-gravado-interface-web)
5. [Modo 3: Analise em Tempo Real](#modo-3-analise-em-tempo-real)
6. [Explicacao das Fases](#explicacao-das-fases)
7. [Interpretacao dos Resultados](#interpretacao-dos-resultados)
8. [Troubleshooting](#troubleshooting)

---

## Instalacao

### Requisitos

- Python 3.9+
- FFmpeg (para composicao de videos)
- iPhone + cabo USB (para modo realtime com Continuity Camera)

### Setup

```bash
# Clonar o repositorio
git clone <repo-url>
cd bowling-cv

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Instalar dependencias
pip install -r requirements.txt

# Verificar instalacao
python -c "import cv2; print(cv2.__version__)"
python -c "import numpy; import pandas; import scipy; print('OK')"
```

### Verificar FFmpeg

```bash
ffmpeg -version
```

Se nao tiver FFmpeg:
```bash
brew install ffmpeg   # macOS
```

---

## Gravacao de Videos

### Posicao da Camera

Para melhores resultados:

- **Angulo**: Camera posicionada **atras do jogador**, capturando a pista inteira do foul line ate os pinos
- **Altura**: Na altura do peito (~1.5m), levemente inclinada para baixo
- **Estabilidade**: Use tripe ou apoie o celular em superficie fixa
- **Enquadramento**: A pista inteira deve estar visivel, com as bordas laterais aparecendo

### Configuracoes do iPhone

- **Resolucao**: 1080p (Full HD) ou superior
- **FPS**: 30fps minimo, 60fps ideal para analise de spin
- **Formato**: MP4 (H.264)
- **Modo**: Video padrao (nao use slow-motion)

### Dicas

- Evite movimentar a camera durante a gravacao
- Iluminacao da bolera e suficiente (nao precisa de luz extra)
- Grave pelo menos 2-3 segundos antes e depois do lancamento
- Recorte o video se necessario (apenas a area da pista)

---

## Modo 1: Analise de Video Gravado (Terminal)

### Pipeline Completo

```bash
# Colocar video em assets/input/
cp meu_video.mp4 assets/input/

# Rodar todas as fases (1, 2, 3, 4, 5)
python main.py --video meu_video.mp4
```

### Fases Individuais

```bash
# Fase 1: Detectar bordas da pista
python main.py --video meu_video.mp4 --phase 1

# Fase 2: Rastrear trajetoria da bola (requer Fase 1)
python main.py --video meu_video.mp4 --phase 2

# Fase 3: Reconstrucao de trajetoria (requer Fase 1 + 2)
python main.py --video meu_video.mp4 --phase 3

# Fase 4: Contar pinos derrubados (requer Fase 1 + 2)
python main.py --video meu_video.mp4 --phase 4

# Fase 5: Analise de spin/rotacao (requer Fase 2)
python main.py --video meu_video.mp4 --phase 5
```

Ou rodar modulos diretamente:
```bash
python -m src.lane_detection.main --video meu_video.mp4
python -m src.ball_detection.main --video meu_video.mp4
python -m src.trajectory_3d.main --video meu_video.mp4
python -m src.pin_detection.main --video meu_video.mp4
python -m src.spin_analysis.main --video meu_video.mp4
```

### Saidas

Os resultados ficam em `output/<nome_video>/`:

```
output/meu_video/
  lane_detection/                   # Phase 1
    boundary_data.json              # Coordenadas das bordas da pista
    final_all_boundaries_*.mp4      # Video com bordas desenhadas
  ball_detection/                   # Phase 2
    trajectory_processed_*.csv      # Coordenadas da bola por frame
    *_ransac_overlay.mp4            # Overlay da bola no video original
    trajectory_on_template.png      # Trajetoria plotada na pista
  trajectory_3d/                    # Phase 3
    transformed_positions.csv       # Trajetoria em coordenadas da pista (106x1829)
    trajectory_overhead.mp4         # Visualizacao top-down
  pin_detection/                    # Phase 4
    *_pin_detection.json            # Contagem de pinos
    *_pin_detection_result.png      # Resultado visual
  spin_analysis/                    # Phase 5
    rotation_data.csv               # Eixo e angulo de rotacao por frame
    rotation_data_processed.csv     # Dados pos-processados
    sphere_video.mp4                # Visualizacao 3D da esfera rotativa
```

---

## Modo 2: Analise de Video Gravado (Interface Web)

### Iniciar

```bash
streamlit run app.py
```

Abre automaticamente no navegador em `http://localhost:8501`.

### Fluxo

1. **Upload**: Arraste ou selecione o video .mp4
2. **Phase 1 - Lane Detection**: Clique "Run" para detectar bordas da pista
3. **Phase 2 - Ball Detection**: Clique "Run" para rastrear a bola
4. **Phase 3 - Reconstruction**: Clique "Run" para reconstruir trajetoria
5. **Phase 4 - Pin Detection**: Clique "Run" para contar pinos
6. **Phase 5 - Spin Analysis**: Clique "Run" para analisar rotacao

Cada fase mostra o video resultado inline no browser.

---

## Modo 3: Analise em Tempo Real

### Setup com iPhone (Continuity Camera)

1. **Requisitos**: macOS Ventura+ e iOS 16+, mesmo Apple ID em ambos
2. **Conexao**: Conecte iPhone ao Mac via **cabo USB** (menor latencia)
3. **Verificar**: O Mac deve reconhecer o iPhone como camera

### Verificar cameras disponiveis

```bash
python -m src.realtime.main --list-cameras
```

Saida esperada:
```
Camera 0: 1280x720 @ 30fps (AVFoundation)    # Webcam do Mac
Camera 1: 1920x1080 @ 30fps (AVFoundation)   # iPhone via Continuity Camera
```

**Nota:** Na primeira vez, o macOS pedira permissao de camera.
Va em System Settings > Privacy & Security > Camera e permita o Terminal.

### Iniciar sessao

```bash
# Camera padrao (webcam do Mac)
python -m src.realtime.main

# iPhone via Continuity Camera (geralmente index 1)
python -m src.realtime.main --camera 1

# Resolucao customizada
python -m src.realtime.main --camera 1 --width 1920 --height 1080
```

### Controles

| Tecla | Acao |
|-------|------|
| Q | Sair |
| R | Recalibrar pista |
| SPACE | Reset para novo lancamento |
| S | Salvar dados da sessao |

### Fluxo da Sessao

1. **Calibracao** (~3s): Aponte a camera para a pista vazia. O sistema coleta 100 frames para detectar as bordas.
2. **Aguardando**: Sistema pronto. Faca o lancamento.
3. **Tracking**: A bola e rastreada automaticamente do foul line ate os pinos.
4. **Pinos**: Apos a bola chegar nos pinos, o sistema conta quantos cairam.
5. **Reset**: Pressione SPACE para o proximo lancamento.

### Posicionamento do iPhone

- Monte o iPhone atras da area de approach
- Angulo: capturando a pista inteira (foul line ate pinos)
- Cabo USB conectado ao Mac
- Use um suporte/tripe para estabilidade

### Dados da sessao

Pressione S para salvar. Os dados ficam em:
```
output/realtime_sessions/session_YYYYMMDD_HHMMSS.json
```

Conteudo:
```json
{
  "timestamp": "20260330_143022",
  "throws": 5,
  "pin_counts": [8, 10, 7, 9, 10],
  "average_pins": 8.8
}
```

---

## Explicacao das Fases

### Phase 1: Lane Detection

Detecta as 4 bordas da pista:
- **Foul line** (horizontal inferior): Canny edges + Hough Lines, filtro por inclinacao
- **Bordas laterais** (verticais): Contornos + Hough Lines, intersecao com foul line
- **Top boundary** (horizontal superior): Sobel edges na regiao dos pinos

Usa um sistema de votacao (bins) para encontrar as linhas mais consistentes em ~100 frames.

### Phase 2: Ball Detection

Rastreia a bola frame a frame:
- **Background subtraction**: MOG2 para detectar movimento
- **ROI tracking**: Area de busca dinamica que acompanha a bola
- **Blob analysis**: Filtros de circularidade, aspect ratio e area
- **Kalman filter**: Predicao de posicao quando a bola nao e detectada
- **Confirmacao**: Precisa de 20 frames consecutivos para confirmar que e a bola (nao a mao)

### Phase 3: Trajectory Reconstruction

Transforma a trajetoria de pixels para coordenadas reais da pista:
- **Homografia per-frame**: Mapeia 4 corners da pista para retangulo padrao (106x1829 px)
- **Perspective transform**: cv2.perspectiveTransform nas posicoes da bola
- **Smoothing**: Savitzky-Golay filter para suavizar a trajetoria
- 106px = largura da pista (~41.5 polegadas / ~105cm)
- 1829px = comprimento da pista (~60 pes / ~18.3m)

### Phase 4: Pin Detection

Conta pinos derrubados via diferenca de frames:
- Compara frame "antes" (pinos em pe) com frame "depois" (apos impacto)
- Diferenca absoluta + threshold + contornos
- Filtros de area e aspect ratio para cada pino

### Phase 5: Spin Analysis

Analisa a rotacao da bola:
- **Optical flow**: Lucas-Kanade na ROI da bola (100 features, window 21x21)
- **Bidirectional check**: Forward + backward tracking, erro < 10px
- **Projecao 3D**: Mapeia pontos 2D para superficie esferica (z = sqrt(r^2 - x^2 - y^2))
- **Kabsch algorithm**: Calcula eixo e angulo de rotacao via SVD decomposition
- **Post-processing**: Remocao de outliers, Gaussian smoothing (sigma=10), interpolacao cubica
- **Visualizacao**: Esfera 3D com padrao xadrez mostrando a rotacao real aplicada

---

## Interpretacao dos Resultados

### Trajetoria (Phase 3)

- **X**: Posicao lateral na pista (0 = esquerda, 106 = direita)
- **Y**: Posicao longitudinal (0 = pinos, 1829 = foul line)
- Trajetoria reta = lancamento reto
- Curva = hook/rotacao lateral

### Spin (Phase 5)

- **x_axis, y_axis, z_axis**: Eixo de rotacao normalizado (vetor unitario)
- **angle**: Angulo de rotacao por frame (radianos)
- **Angular velocity**: angle * FPS (rad/s)
- **RPM estimado**: (angular_velocity / (2 * pi)) * 60

**Como interpretar a esfera 3D:**
- A seta vermelha indica a direcao do eixo de rotacao
- O padrao xadrez mostra a rotacao real aplicada
- Seta para cima = forward roll (bola rolando para frente)
- Seta lateral = side rotation (hook/curva)
- Valores tipicos de RPM: iniciantes 100-200, avancados 300-500

### Pinos (Phase 4)

- Contagem de 0 a 10 pinos derrubados
- Strike = 10 pinos no primeiro lancamento
- Spare = derrubar todos os restantes no segundo lancamento

---

## Troubleshooting

### "No cameras found"
- Verifique que o iPhone esta conectado via USB
- Verifique que Bluetooth e WiFi estao ligados
- Tente desconectar e reconectar o cabo
- Verifique que ambos dispositivos usam o mesmo Apple ID

### "Could not open video"
- Verifique que o video esta em `assets/input/`
- Verifique que o formato e MP4 (H.264)
- Converta se necessario: `ffmpeg -i video.mov -c:v libx264 video.mp4`

### "Lane detection failed"
- A pista inteira precisa estar visivel no video
- Verifique iluminacao (sem reflexos fortes no chao)
- Tente ajustar a camera para um angulo mais reto

### "Ball not detected"
- A bola precisa ter contraste com a pista
- Verifique que o fundo esta relativamente estatico
- Ajuste `MOG2_VAR_THRESHOLD` no config (menor = mais sensivel)

### "Spin analysis has too few points"
- Precisa de pelo menos 3 features detectadas na bola por frame
- Bolas lisas (sem marca/padrao) tem menos features
- Aumente MAX_CORNERS no config de spin ou reduza QUALITY_LEVEL

### FFmpeg errors
- Instale: `brew install ffmpeg`
- Verifique: `ffmpeg -version`

### Import errors
- Verifique que esta no diretorio raiz do projeto
- Ative o venv: `source .venv/bin/activate`
- Reinstale: `pip install -r requirements.txt`

### Performance lenta no modo real-time
- Reduza resolucao: `--width 640 --height 480`
- Feche outros programas pesados
- Use cabo USB (nao WiFi) para Continuity Camera

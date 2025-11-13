# NeuroWaves

Plataforma interativa para classificar ondas cerebrais (Delta, Theta, Alpha, Beta e Gamma) a partir de sinais vindos de um EEG/EMG ligado via cabo P2 ou de gravações já existentes em formato de áudio.

## Destaques
- Interface redesenhada em duas telas: painel offline com espectrograma em tela cheia + trilha colorida, e monitor ao vivo dedicado.
- Captura ao vivo via qualquer entrada de áudio reconhecida pelo sistema (line-in, microfone dedicado, interface USB, etc.).
- Upload de arquivos `.wav`, `.flac`, `.ogg`, `.mp3` **e** `.mkv` (com extração automática via MoviePy/ffmpeg), com transporte completo: play/pause, reinício, retroceder 5s e scrub interativo.
- Visualização avançada com espectrograma estabilizado e trilha temporal que pinta cada janela conforme a banda dominante (Delta/Theta/Alpha/Beta/Gamma) enquanto o áudio avança.
- Indicadores em tempo real para os dois modos (badge colorido + barras de energia), além da lista das principais frequências e histórico consolidado.
- Controle de zoom espectral para destacar apenas as frequências relevantes (ex.: até 60 Hz para EEG).
- Classificador heurístico baseado em análise espectral (FFT) com normalização automática do sinal e janela de Hann.

## Pré-requisitos
1. Python 3.10 ou superior.
2. Dependências listadas em `requirements.txt`:
   ```bash
   python -m pip install -r requirements.txt
   ```
   > Observações:
   > - `sounddevice` se conecta diretamente a bibliotecas de áudio do sistema (PortAudio). Caso esteja no Windows, o instalador oficial do Python já traz o binário necessário.
   > - Para carregar `.mkv` é preciso ter `moviepy` e um `ffmpeg` acessível via PATH. Consulte `DOCUMENTATION.md` para validação passo a passo.

## Como executar
```bash
python main.py
```

## Fluxo sugerido
1. **Classificação de arquivos (aba “Análise Offline”)**
   - Clique em *“Carregar gravação EEG / áudio”* e selecione o arquivo.
   - Use os botões Play/Pause/Stop, Reiniciar, Retroceder 5s e o slider de scrub para navegar; o espectrograma e as barras mudam de cor conforme a banda dominante.
   - Consulte o painel de frequências para descobrir os picos dominantes, acompanhe o histórico e verifique o badge colorido para saber a banda “ao vivo”.
   - Ajuste o controle “Zoom espectral” para limitar o eixo vertical do espectrograma e focar nas bandas de interesse.
2. **Monitoramento ao vivo (aba “Monitor Ao Vivo”)**
   - Conecte seu EEG/EMG na entrada P2 (ou outro dispositivo de áudio reconhecido pelo sistema).
   - Escolha o dispositivo e configure taxa de amostragem e janela de análise (a mesma janela influencia a segmentação offline).
   - Clique em *“Iniciar Monitoramento”* para liberar a tela dedicada; o badge colorido indica instantaneamente a banda dominante.

## Arquitetura em alto nível
- `SignalProcessor`: higieniza o sinal, calcula espectro com FFT (janela de Hann), gera espectrogramas e lista as frequências mais energéticas.
- `LiveStreamWorker`: roda em `QThread`, coleta blocos do `sounddevice.InputStream` e abastece o monitor ao vivo.
- `FilePlaybackWorker`: reproduz arquivos com `sounddevice.OutputStream`, sincronizando áudio, marcador móvel, retrocesso e scrub.
- `WavePlotCanvas`: componente Matplotlib configurável (com ou sem forma-de-onda) que agrega espectrograma, trilha colorida e barras de energia.
- `NeuroWavesWindow`: organiza as duas abas, coordena o transporte (play/pause/seek), pré-calcula segmentos e mantém as métricas/histórico.

> Para um guia minucioso (atalhos, detalhes da UI, troubleshooting), consulte `DOCUMENTATION.md`.

## Próximos passos (opcionais)
1. Persistir classificações em banco de dados para auditoria.
2. Treinar e integrar um modelo de machine learning com dados reais de EEG para aumentar a assertividade.
3. Adicionar exportação de relatórios (PDF/CSV) diretamente da interface.

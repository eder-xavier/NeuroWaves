# NeuroWaves · Documentação Completa

## 1. Visão geral
NeuroWaves é um painel desktop escrito em Python/Qt para classificar bandas cerebrais (Delta, Theta, Alpha, Beta e Gamma) em duas frentes:
- **Análise Offline:** abre gravações (`.wav`, `.flac`, `.ogg`, `.mp3`, `.mkv`) e reproduz o áudio com controles avançados (play/pause, reinício, retroceder 5 s e scrub). O espectrograma ocupa a maior parte da tela, enquanto uma trilha temporal pinta a banda dominante de cada janela antes mesmo de iniciar a reprodução.
- **Monitoramento Ao Vivo:** permite conectar um EEG/EMG (via P2, interface USB ou placa de som) e visualizar em tempo real a banda dominante, barras de energia e histórico de diagnósticos.

## 2. Instalação passo a passo
1. **Pré-requisitos**
   - Python 3.10+ (64 bits recomendado).
   - PortAudio (já incluso no instalador oficial do Python para Windows).
   - ffmpeg no `PATH` (necessário apenas para `.mkv`).
2. **Instalar dependências**
   ```bash
   python -m pip install -r requirements.txt
   ```
3. **Validar o suporte a MKV (opcional)**
   ```bash
   python - <<'PY'
   import moviepy, shutil
   print('moviepy:', moviepy.__version__)
   print('ffmpeg:', shutil.which('ffmpeg'))
   PY
   ```
   Se `ffmpeg` retornar `None`, instale-o e adicione o diretório `bin` ao `PATH`.

## 3. Estrutura do app
```
NeuroWaves/
├─ main.py              # aplicação PyQt5 completa
├─ README.md            # resumo rápido
├─ DOCUMENTATION.md     # este guia detalhado
└─ requirements.txt     # dependências
```

## 4. Aba “Análise Offline”
### 4.1. Controles
- **Carregar gravação:** abre arquivos `.wav`, `.flac`, `.ogg`, `.mp3` e `.mkv`. Para MKV, o áudio é extraído pelo MoviePy.
- **Transporte:**
  - `▶ Reproduzir`: inicia do ponto atual ou retoma se estiver pausado.
  - `⏸ Pausar`: congela o áudio mantendo o ponto atual.
  - `⏹ Parar`: encerra a reprodução e volta para 00:00.
  - `⏪ -5s`: retrocede cinco segundos.
  - `⏮ Reiniciar`: volta instantaneamente ao início.
- **Slider de Scrub:** representa a linha do tempo (0–100%). Arraste para pular para qualquer ponto; ao soltar, o sinal é recalculado e, se estava tocando, a reprodução continua do novo ponto.
- **Zoom Espectral:** defina o limite superior (20–200 Hz) para o eixo vertical do espectrograma. Ideal para focar nas bandas clássicas de EEG (Delta–Gamma).
- **Indicadores:**
  - Badge colorido “Ao vivo · Banda” mostra a banda dominante do trecho atual.
  - Labels exibem frequência dominante e energia RMS.
  - Lista “Principais frequências” apresenta os picos espectrais da gravação completa.
  - Histórico registra cada janela classificada (offline ou ao vivo) com carimbo de hora.

### 4.2. Visualização
- **Espectrograma:** gerado uma única vez (FFT com janela Hann) e mantido na memória. A cada janela reproduzida, o trecho atual é destacado com uma faixa translúcida e o cursor amarelo acompanha o tempo.
- **Trilha temporal:** pré-calculada usando a mesma janela configurada no campo “Janela de análise”. Cada segmento recebe uma cor (Delta/Theta/Alpha/Beta/Gamma). Assim você enxerga a distribuição das bandas antes mesmo de apertar play.
- **Barras de energia:** exibem a porcentagem de energia da janela atual.

### 4.3. Ajustando a janela
O campo “Janela de análise” (na aba ao vivo) controla também a segmentação offline. Ao alterar esse valor, a aplicação pausa a reprodução, recalcula toda a trilha colorida e redesenha o espectrograma de referência automaticamente.

### 4.4. Zoom espectral
O seletor “Zoom espectral (limite superior)” altera o eixo Y do espectrograma offline. Exemplos:
- 60 Hz: foco em EEG clássico (Delta–Gamma).
- 120 Hz: monitora harmônicos/artefatos musculares.

Sempre que alterar o zoom, o espectrograma completo é redesenhado mantendo o cursor/realce atual.

## 5. Aba “Monitor Ao Vivo”
1. Escolha o dispositivo de entrada listado (qualquer dispositivo com canais de captura).
2. Ajuste taxa de amostragem e janela de análise.
3. Clique em **Iniciar Monitoramento** para abrir o fluxo contínuo. O badge e as barras de energia são atualizados bloco a bloco.
4. **Parar** encerra o `InputStream` e libera o dispositivo.

## 6. Pipeline de classificação
1. **Pré-processamento:** remoção da média, normalização pelo desvio padrão e aplicação de janela de Hann.
2. **FFT:** cálculo do espectro de potência (`np.abs(rfft(x)) ** 2`).
3. **Energia por banda:** média dos bins dentro de cada faixa (Delta 0.5–4 Hz, Theta 4–8 Hz, Alpha 8–13 Hz, Beta 13–30 Hz, Gamma 30–100 Hz).
4. **Banda dominante:** a maior energia normalizada define o rótulo e a cor exibida no badge, timeline, barras e espectrograma.

## 7. Mapas de cor
| Banda | Faixa (Hz) | Cor |
|-------|------------|-----|
| Delta | 0.5–4      | #00bcd4 |
| Theta | 4–8        | #4caf50 |
| Alpha | 8–13       | #ffc107 |
| Beta  | 13–30      | #ff5722 |
| Gamma | 30–100     | #9c27b0 |

Essas cores são usadas em todo o app (badge, barras, trilha e espectrograma preenchido).

## 8. Resolução de problemas
| Sintoma | Causa provável | Como resolver |
|---------|----------------|----------------|
| `Import "moviepy.editor" could not be resolved` | MoviePy não instalado no mesmo ambiente do VS Code | Rode `python -m pip install moviepy` e selecione o mesmo interpretador no IDE. |
| Falha ao abrir `.mkv` | ffmpeg ausente do `PATH` | Baixe ffmpeg, extraia e adicione `.../ffmpeg/bin` ao `PATH`, depois reinicie o terminal. |
| Nenhum dispositivo aparece ao vivo | Permissões de áudio ou driver indisponível | Verifique se o microfone/line-in está habilitado no sistema e clique em “Atualizar dispositivos”. |
| Slider não move a reprodução | Arquivo não carregado ou duração zero | Carregue um arquivo válido; o slider permanece bloqueado até que a duração seja conhecida. |
| Interface congelando com arquivos longos | Janela de análise muito pequena (muitas janelas) | Aumente a “Janela de análise” para reduzir o número de segmentos pré-calculados. |

## 9. Dicas finais
- Use fones/capturas isoladas para reduzir ruído na classificação.
- Ajuste a janela para balancear resolução temporal (janelas curtas) x estabilidade (janelas longas).
- Para auditorias, exporte o histórico da aba offline (Ctrl+C no item selecionado) e cole onde preferir.
- Execute `python3 -m py_compile main.py` sempre que alterar o código para garantir que a interface sobe sem erros de sintaxe.

Bom uso! Qualquer melhoria adicional pode ser discutida diretamente no arquivo `main.py`, especialmente dentro das classes `SignalProcessor`, `FilePlaybackWorker` e `NeuroWavesWindow`.

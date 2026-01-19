# Alignment with GRPO

> **Все обучение производилось на NVIDIA H100**

## Структура:
    1) scripts -> папка со скриптами:
        1.1) grpo_train.py - общий пайплайн GRPO обучения + различные rule-based reward функции (в том числе и LCPO)
        1.2) sft_grpo_train.py - скрипт GRPO обучения с lora весами от предобученной SFT модели (на gsm8k)
    2) notebooks -> папка с ноутбуками, которые можно запускать в google colab:
        2.1) grpo_train.ipynb - общий пайплайн GRPO обучения в виде блокнота
        2.2) sft_before_grpo.ipynb - скрипт SFT на датасете gsm8k, lora веса после обучения нужно сохранить в папке scripts/lora_after_sft/ для дальнейшего GRPO в sft_grpo_train.py 
        2.3) change_len_prompt_bert.ipynb - обучение ModernBERT для регрессии количества токенов в ответе по входному запросу и запуск бейзлайна с новым промптом
    3) logs -> папка с логами обучения каждого эксперимента
    4) REPORT.md -> отчет о проделанной работе

## Требования для запуска скриптов:
#### Если вы решили использовать скрипты, то нужно:
    1) python ">=3.11,<3.13" версий
    2) linux
    3) CUDA Toolkit: 12.4

## Зависимости:
#### Перед запуском скриптов нужно сделать:
    1) sudo apt update
    2) sudo apt install -y build-essential cmake git ninja-build libomp-dev libopenblas-dev
    3) python3 -m venv .venv
    4) source .venv/bin/activate
    5) pip install --upgrade pip setuptools wheel
    6) pip install -r requirements.txt
#### Теперь можно спокойно запускать скрипты!

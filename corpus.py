#!/usr/bin/env python3
"""
Diverse local corpus, packed into fixed-length token windows.

The activation-subspace question is "where does k saturate on a DIVERSE
corpus", and the earlier measurement (3 prompts, 420 samples, k=320) could not
answer it: with 420 samples a 320-dimensional basis is nearly rank-saturated by
construction, so 98% retention was guaranteed rather than measured. Fixing that
needs (a) far more samples than k and (b) documents held out from the fit.

Everything here is already on this machine -- no download. Genres are kept
separate so the analysis can do both splits that matter: held-out *documents*
within a genre, and leave-one-*genre*-out, which is the real test of whether one
offline basis transfers to text it was never fitted on.

Each window is a pack of several documents from one genre, truncated/joined to
exactly SEQ tokens. Packing matters: a window is one prefill, and one prefill of
this model reads ~64 GB off the SSD regardless of length, so long windows are
the only affordable way to collect tens of thousands of samples.
"""

import glob
import json
import os
import random
from pathlib import Path

SEQ = 1024
HF = os.path.expanduser("~/.cache/huggingface/datasets")


def _arrow(pattern):
    import pyarrow as pa
    hits = glob.glob(os.path.join(HF, pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return pa.ipc.open_stream(pa.memory_map(hits[0])).read_all()


def wiki_docs():
    t = _arrow("Salesforce___wikitext/*/*/*/wikitext-train.arrow")
    out, cur = [], []
    for s in t.column("text").to_pylist():
        if s.startswith(" = ") and not s.startswith(" = = "):
            if len(cur) > 8:
                out.append("".join(cur))
            cur = []
        cur.append(s)
    return out


def mmlu_docs(stem=True):
    STEM = {"abstract_algebra", "anatomy", "astronomy", "college_biology",
            "college_chemistry", "college_computer_science", "college_mathematics",
            "college_physics", "computer_security", "conceptual_physics",
            "electrical_engineering", "elementary_mathematics", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_mathematics", "high_school_physics", "high_school_statistics",
            "machine_learning", "econometrics", "formal_logic"}
    t = _arrow("cais___mmlu/all/*/*/mmlu-test.arrow")
    q = t.column("question").to_pylist()
    s = t.column("subject").to_pylist()
    c = t.column("choices").to_pylist()
    a = t.column("answer").to_pylist()
    out = []
    for qi, si, ci, ai in zip(q, s, c, a):
        if (si in STEM) != stem:
            continue
        opts = "\n".join(f"{chr(65+j)}. {o}" for j, o in enumerate(ci))
        out.append(f"Subject: {si.replace('_', ' ')}\nQuestion: {qi}\n{opts}\n"
                   f"Answer: {chr(65+ai)}\n")
    return out


def dolly_docs():
    t = _arrow("databricks___databricks-dolly-15k/*/*/*/*.arrow")
    i = t.column("instruction").to_pylist()
    c = t.column("context").to_pylist()
    r = t.column("response").to_pylist()
    return [f"### Instruction\n{a}\n\n### Context\n{b}\n\n### Response\n{d}\n"
            for a, b, d in zip(i, c, r)]


def qanta_docs():
    t = _arrow("community-datasets___qanta/*/*/*/qanta-buzztest.arrow")
    return [s for s in t.column("text").to_pylist() if len(s) > 200]


def swebench_docs():
    docs = []
    for f in ("princeton-nlp___swe-bench_lite/*/*/*/*.arrow",
              "PGCodeLLM___feat_bench/*/*/*/*.arrow"):
        t = _arrow(f)
        names = t.schema.names
        for col in ("problem_statement", "patch", "test_patch", "hints_text"):
            if col in names:
                docs += [s for s in t.column(col).to_pylist() if s and len(s) > 300]
    return docs


def code_docs():
    """Real source from this machine: MLX, mlx_lm, and this project."""
    roots = [os.path.dirname(os.path.abspath(__file__))]
    import mlx.core, mlx_lm, numpy
    roots += [os.path.dirname(mlx_lm.__file__), os.path.dirname(numpy.__file__),
              os.path.dirname(os.path.dirname(mlx.core.__file__))]
    docs = []
    for r in roots:
        for p in sorted(Path(r).rglob("*.py"))[:400]:
            try:
                s = p.read_text()
            except Exception:
                continue
            if len(s) > 600:
                docs.append(f"# {p.name}\n{s}")
    for r in roots:
        for pat in ("*.cpp", "*.h", "*.metal"):
            for p in sorted(Path(r).rglob(pat))[:60]:
                try:
                    s = p.read_text()
                except Exception:
                    continue
                if len(s) > 600:
                    docs.append(f"// {p.name}\n{s}")
    return docs


MULTILING = [
"""Les modèles de mélange d'experts posent un problème particulier sur une
machine dont la mémoire est limitée. Chaque jeton n'active qu'une petite partie
des poids, mais on ignore laquelle avant d'avoir calculé le routeur, et le
routeur dépend de l'état caché produit par la couche précédente. Autrement dit,
la connaissance arrive toujours trop tard pour être utile, sauf si l'on accepte
de deviner. C'est exactement ce que fait la spéculation entre couches: on
applique le routeur de la couche suivante à l'état caché de la couche courante,
et l'on obtient une prédiction gratuite, sans entraînement ni poids
supplémentaires. Le flux résiduel change lentement d'une couche à l'autre, donc
les entrées du routeur se ressemblent, et la prédiction est correcte environ
neuf fois sur dix. Ce qui reste, ce sont les lectures manquées, et celles-ci
bloquent le calcul.

La mémoire unifiée d'Apple complique encore les choses. Le processeur graphique
et le processeur central partagent la même bande passante, si bien qu'un accès
au disque ne se paie pas seulement en latence mais aussi en contention. Une
lecture de deux mégaoctets et demi coûte peu si elle est lancée assez tôt; la
même lecture, demandée au moment où le calcul en a besoin, coûte tout le temps
qu'elle prend. La conception se réduit donc à une question d'ordonnancement.""",

"""El problema de ejecutar un modelo de sesenta y cinco gigabytes en una máquina
de dieciséis gigabytes no es realmente un problema de memoria, sino de ancho de
banda. Si cada token exige leer mil megabytes del disco, y el disco entrega tres
gigabytes por segundo en el mejor de los casos, entonces la velocidad máxima
está fijada de antemano y ninguna optimización del núcleo puede superarla. La
única salida consiste en leer menos bytes. Hay tres maneras de conseguirlo:
guardar menos expertos, guardar cada experto con menos bits, o descubrir que la
matriz de pesos nunca se usa por completo.

La tercera vía es la más interesante porque no es una compresión de los pesos
sino de las activaciones. Un producto entre una matriz y un vector solo necesita
la parte de la matriz que el vector realmente visita. Si los vectores de entrada
viven en un subespacio de doscientas o trescientas dimensiones dentro de un
espacio de casi tres mil, entonces el resto de la matriz nunca contribuye nada,
y almacenarlo es puro desperdicio. La pregunta empírica es si ese subespacio
sigue siendo pequeño cuando el texto es variado.""",

"""Der entscheidende Punkt bei speicherbeschränkter Inferenz ist, dass die
Gesamtzahl der gelesenen Bytes pro Token die Obergrenze der Geschwindigkeit
festlegt. Alles andere, Kernel, Zeitpläne, Warteschlangen, verschiebt lediglich
Arbeit innerhalb dieser Grenze. Wer die Grenze verschieben will, muss die Menge
der Daten verringern, die überhaupt bewegt werden.

Es gibt zwei Arten von Redundanz in einem quantisierten Mixture-of-Experts-
Modell. Die erste liegt in den Gewichten selbst und ist nach der
Vier-Bit-Quantisierung weitgehend erschöpft: die Experten sind dicht, besitzen
vollen Rang und stehen fast orthogonal zueinander. Die zweite liegt in den
Aktivierungen, und sie ist noch offen. Wenn alle Experten einer Schicht dieselbe
Basis teilen, dann kostet die Basis fast nichts, und jeder Experte speichert nur
seine Projektion. Ob das funktioniert, hängt allein davon ab, wie schnell die
erklärte Varianz mit der Dimension sättigt und ob eine offline berechnete Basis
auf neuen Text überträgt.""",

"""在一台只有十六吉字节内存的笔记本电脑上运行一个六百五十亿参数的混合专家模型，
真正的限制并不是内存容量，而是每个词元需要从固态硬盘读取的字节数。如果每个词元
需要读取一千兆字节，而硬盘的峰值带宽是每秒三点四吉字节，那么速度上限就已经被
确定了，任何内核优化都无法突破它。唯一的办法是减少读取的字节数。

权重空间的冗余已经被四比特量化基本耗尽：专家矩阵是稠密的、满秩的，而且彼此近乎
正交。剩下的机会在激活空间。矩阵与向量相乘时，只有向量真正到达的子空间才起作用。
如果隐藏状态实际上只在两三百个方向上变化，而模型的宽度是两千八百八十，那么其余
方向上的权重从未被使用过，把它们从磁盘读进来纯粹是浪费。关键问题在于，当文本足够
多样时，这个子空间是否仍然很小，以及在一种语料上拟合的基是否能迁移到另一种语料。""",

"""メモリの制約が厳しい環境で大規模な混合エキスパートモデルを動かすとき、
本当のボトルネックは計算ではなく、トークンごとに読み込むバイト数である。
毎トークンおよそ一ギガバイトを読む必要があり、記憶装置の帯域が毎秒三ギガ
バイト程度であれば、速度の上限はその時点で決まってしまう。カーネルをいくら
最適化しても、読まなければならないバイトを掛け算することはできない。

したがって残された道は三つある。常駐させるエキスパートを増やすこと、各
エキスパートのビット数を減らすこと、そして重み行列のうち実際に使われる部分
だけを保存することである。三番目は重みの圧縮ではなく活性の圧縮であり、同じ層
の全てのエキスパートが一つの基底を共有できるという点で特に安価である。問題は、
多様な文章を与えたときにその部分空間が本当に小さいままかどうかだ。""",

"""Основное ограничение при выполнении очень большой модели на машине с малым
объёмом памяти заключается не в вычислениях, а в количестве байтов, читаемых с
диска на каждый токен. Если для одного токена требуется прочитать около
гигабайта, а накопитель выдаёт три с половиной гигабайта в секунду, то
предельная скорость определена заранее.

Избыточность в пространстве весов после четырёхбитного квантования практически
исчерпана: матрицы экспертов плотные, полного ранга и почти ортогональны друг
другу. Однако в пространстве активаций избыточность остаётся. Произведение
матрицы на вектор использует лишь ту часть матрицы, которую вектор
действительно посещает, и если скрытые состояния занимают подпространство
размерности в несколько сотен, остальное можно не хранить вовсе.""",

"""Il vero limite nell'esecuzione di un modello enorme su una macchina piccola
non è la capacità di calcolo ma la quantità di byte letti per ogni token. Se
ogni token richiede la lettura di un gigabyte e il disco fornisce tre gigabyte
al secondo, la velocità massima è già decisa. Ridurre i byte è l'unica strada
che conta davvero, e la compressione delle attivazioni è quella meno esplorata.

O ponto essencial é o mesmo em português: uma multiplicação entre matriz e vetor
só precisa da parte da matriz que o vetor realmente visita. Se os estados
ocultos vivem num subespaço de algumas centenas de dimensões, o restante da
matriz nunca contribui. A questão empírica é se esse subespaço continua pequeno
quando o texto é variado, e se uma base calculada offline transfere para texto
que ela nunca viu.""",

"""ऐसे मॉडल को चलाने में असली बाधा गणना नहीं है, बल्कि हर टोकन के लिए डिस्क से पढ़े जाने वाले
बाइट्स की संख्या है। यदि प्रत्येक टोकन के लिए लगभग एक गीगाबाइट पढ़ना पड़े और डिस्क प्रति सेकंड
साढ़े तीन गीगाबाइट ही दे सके, तो गति की ऊपरी सीमा पहले से तय हो जाती है।

चार-बिट परिमाणीकरण के बाद भार-स्थान में बची हुई अतिरेकता लगभग समाप्त हो चुकी है। शेष अवसर
सक्रियण-स्थान में है: आव्यूह-सदिश गुणन में केवल वही दिशाएँ मायने रखती हैं जिन तक सदिश वास्तव
में पहुँचता है। यदि छिपी हुई अवस्थाएँ कुछ सौ दिशाओं में ही रहती हैं, तो बाकी भार कभी उपयोग में
नहीं आते।""",

"""إن القيد الحقيقي عند تشغيل نموذج ضخم على جهاز محدود الذاكرة ليس الحساب، بل عدد
البايتات التي تُقرأ من القرص لكل رمز. فإذا احتاج كل رمز إلى قراءة نحو غيغابايت
واحد، وكان القرص يوفر ثلاثة غيغابايت في الثانية، فإن الحد الأعلى للسرعة يكون
محدداً سلفاً، ولا يمكن لأي تحسين في النواة أن يتجاوزه.

لقد استُنفدت تقريباً الزوائد في فضاء الأوزان بعد التكميم إلى أربع بتات. أما فضاء
التنشيط فلا يزال مفتوحاً: حاصل ضرب مصفوفة في متجه لا يحتاج إلا إلى الجزء الذي
يزوره المتجه فعلاً من المصفوفة.""",
]


def multiling_docs():
    return MULTILING * 3


GENRES = {
    "wiki": wiki_docs,
    "mmlu_stem": lambda: mmlu_docs(stem=True),
    "mmlu_hum": lambda: mmlu_docs(stem=False),
    "dolly": dolly_docs,
    "qanta": qanta_docs,
    "swebench": swebench_docs,
    "code": code_docs,
    "multiling": multiling_docs,
}


def build(tokenizer, per_genre=3, seq=SEQ, seed=0):
    """Return [(genre, window_index, [token ids] of length seq)]."""
    rng = random.Random(seed)
    windows = []
    for gname, fn in GENRES.items():
        docs = fn()
        rng.shuffle(docs)
        need = per_genre
        cur, wi = [], 0
        for d in docs:
            cur += tokenizer.encode(d)
            cur.append(tokenizer.eos_token_id if tokenizer.eos_token_id is not None
                       else 0)
            while len(cur) >= seq and wi < need:
                windows.append((gname, wi, cur[:seq]))
                cur = cur[seq:]
                wi += 1
            if wi >= need:
                break
        if wi < need:
            raise RuntimeError(f"genre {gname}: only {wi}/{need} windows")
    return windows


if __name__ == "__main__":
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    tok = load_tokenizer(Path(os.path.expanduser("~/Desktop/moe-stream/model-120b")))
    w = build(tok, per_genre=int(os.environ.get("PER_GENRE", "3")))
    print(f"{len(w)} windows x {SEQ} tokens = {len(w)*SEQ:,} samples/layer")
    for g in GENRES:
        n = sum(1 for x in w if x[0] == g)
        ex = next(x for x in w if x[0] == g)
        print(f"  {g:<10} {n} windows   first 60 chars: "
              f"{tok.decode(ex[2][:24])[:60]!r}")
    json.dump([[g, i, t] for g, i, t in w],
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "corpus_windows.json"), "w"))

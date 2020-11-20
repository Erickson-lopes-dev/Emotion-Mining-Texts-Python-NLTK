import nltk

# nltk.download()

base = [('eu sou admirada por muitos', 'alegria'),
        ('me sinto completamente amado', 'alegria'),
        ('amar e maravilhoso', 'alegria'),
        ('estou me sentindo muito animado novamente', 'alegria'),
        ('eu estou muito bem hoje', 'alegria'),
        ('que belo dia para dirigir um carro novo', 'alegria'),
        ('o dia está muito bonito', 'alegria'),
        ('estou contente com o resultado do teste que fiz no dia de ontem', 'alegria'),
        ('o amor e lindo', 'alegria'),
        ('nossa amizade e amor vai durar para sempre', 'alegria'),

        ('estou amedrontado', 'medo'),
        ('ele esta me ameacando a dias', 'medo'),
        ('isso me deixa apavorada', 'medo'),
        ('este lugar e apavorante', 'medo'),
        ('se perdermos outro jogo seremos eliminados e isso me deixa com pavor', 'medo'),
        ('tome cuidado com o lobisomem', 'medo'),
        ('se eles descobrirem estamos encrencados', 'medo'),
        ('estou tremendo de medo', 'medo'),
        ('eu tenho muito medo dele', 'medo'),
        ('estou com medo do resultado dos meus testes', 'medo')]

stopwords = ['a', 'agora', 'algum', 'alguma', 'aquele', 'aqueles', 'de', 'deu', 'do', 'e', 'estou', 'esta', 'esta',
             'ir', 'meu', 'muito', 'mesmo', 'no', 'nossa', 'o', 'outro', 'para', 'que', 'sem', 'talvez', 'tem', 'tendo',
             'tenha', 'teve', 'tive', 'todo', 'um', 'uma', 'umas', 'uns', 'vou']

# print(base[1])

stopwordsnltk = nltk.corpus.stopwords.words('portuguese')  # lista personalizada com o padrão de stopwords do nltk


# print(stopwordsnltk)


def removerstoprwords(texto):
    frases = []
    for (palavras, emocao) in texto:
        # retira os stopwords e coloca apenas as palavras
        semstop = [p for p in palavras.split() if p not in stopwordsnltk]
        frases.append((semstop, emocao))
    return frases


# print(removerstoprwords(base))


# podemos perder um pouco de informação com stemming
def aplicastemmer(texto):
    # stem expecifico para portugues
    stemmer = nltk.stem.RSLPStemmer()
    frasesstemmer = []

    for (palavras, emocao) in texto:
        # encontrar o radical de cada palavra e retirando as stopwords
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwordsnltk]
        frasesstemmer.append((comstemming, emocao))

    return frasesstemmer


frasescomstemming = aplicastemmer(base)
# print(frasescomstemming)


def buscapalavras(frases):
    todaspalavras = []
    for (palavras, emocao) in frases:
        # jogar somente as palavras sem as emoções
        todaspalavras.extend(palavras)
    return todaspalavras


palavras = buscapalavras(frasescomstemming)

# print(palavras)


def buscafrequencia(palavras):
    # mostra a frequencia das palavras
    palavras = nltk.FreqDist(palavras)
    return palavras


frequencia = buscafrequencia(palavras)

# 50 primeiras palavras
# print(frequencia.most_common(50))


def buscapalavrasunicas(palavras):
    # keys para retornar as chaves de cada um sem repetições
    freq = frequencia.keys()
    return freq


palavrasunicas = buscapalavrasunicas(frequencia)

# print(palavrasunicas)


def extratorpalavras(documento):
    # tornar cada palavra unica
    doc = set(documento)
    caracteristicas = {}

    for palavras in palavrasunicas:
        caracteristicas[f'{palavras}'] = (palavras in doc)
    return caracteristicas


caracteristicasfrases = extratorpalavras(['am', 'nov', 'dia'])

# utilizando algoritmo de stemming
# print(caracteristicasfrases)

# aplicar caracteristicas / passa a função que faz a extração de cada palavra
# passando as frases com stemming ( separação por radical) e emoção da frase
# verifica se tem a especifica caracteristica para ser frase de medo ou alegria
basecompleta = nltk.classify.apply_features(extratorpalavras, frasescomstemming)

# print(basecompleta[10])

# consatrpoi a tablea de probabilidade
classificador = nltk.NaiveBayesClassifier.train(basecompleta)

# exibe os classificadores
# print(classificador.labels())

# exibe as palavras mais informativas e suas probabilidades
# print(classificador.show_most_informative_features(10))

teste = 'eu te amo amor'

testestemming = []
stemmer = nltk.stem.RSLPStemmer()
# percorre as palavras
for palavras in teste.split():
    # joga cada uma das palavras na var
    comstem = [p for p in palavras.split()]
    # pega o radical
    testestemming.append(str(stemmer.stem(comstem[0])))

print(testestemming)

novo = extratorpalavras(testestemming)

print(novo)

print(classificador.classify(novo))

distribuicaoo = classificador.prob_classify(novo)

for classe in distribuicaoo.samples():
    print(f'{classe:>7} {distribuicaoo.prob(classe)}')
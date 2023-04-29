from __future__ import annotations

import math
import numpy

from matplotlib import pyplot
from api.regras.iaRegras import DatasetRNN

class DBN:
    def __init__(self, nCamadaVisivel: int, nCamadaOculta: int, txAprendizado: float):
        self.nCamadaVisivel = nCamadaVisivel
        self.nCamadaOculta = nCamadaOculta
        self.txAprendizado = txAprendizado
        self.pesos = numpy.random.uniform(-1,1,(nCamadaVisivel, nCamadaOculta))
        self.biasVisivel = numpy.zeros(nCamadaVisivel)
        self.biasOculto = numpy.zeros(nCamadaOculta)
        self.estados_ocultos = []
        self.is_camada_saida = False


    def sigmoid(self, x):
        sig = 1 / (1 + numpy.exp(-x))
        return sig

    def derivada_sigmoid(self, x):
        dsig = x * (1 - x)
        return dsig

    def softmax(self, x):
        exp_puntuacao = numpy.exp(x, dtype=numpy.float64)
        return exp_puntuacao / numpy.sum(exp_puntuacao, axis=0, dtype=numpy.float64)

    def derivada_softmax(self, x):
        s = self.softmax(x)
        ds = numpy.zeros_like(s)
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                ds[i, j] = s[i, j] * (numpy.identity(s.shape[0])[i, j] - s[:, j]).sum()
        return ds

    def amostraOculta(self, entradas: list):
        '''
            propaga a entrada para a saida com base nos pesos globais da classe e o bias oculta

            @return
            probabilidades_ocultas, estados_ocultos
        '''
        entradasNormalizadas = numpy.asarray(entradas)
        ativacoes_ocultas = numpy.dot(entradasNormalizadas, self.pesos) + self.biasOculto
        probabilidades_ocultas = self.sigmoid(ativacoes_ocultas)
        estados_ocultos = numpy.random.binomial(1, probabilidades_ocultas)

        return probabilidades_ocultas, estados_ocultos


    def amostraVisivel(self, saidas: list):
        '''
            Reconstroi a entrada com base nas saidas.

            @return
            probabilidades_visiveis, estados_visiveis
        '''
        pesosTranspostos = numpy.transpose(self.pesos)
        ativacoes_visiveis = numpy.dot(saidas, pesosTranspostos) + self.biasVisivel
        probabilidades_visiveis = self.sigmoid(ativacoes_visiveis)
        estados_visiveis = numpy.random.binomial(1, probabilidades_visiveis)

        return probabilidades_visiveis, estados_visiveis


    def cross_entropia(self, y_true, y_pred):
        y_true_normalizado = numpy.asarray(y_true)
        y_pred_normalizado = numpy.asarray(y_pred)
        loss = -numpy.mean(y_true_normalizado * numpy.log(y_pred_normalizado + 1e-8) + (1 - y_true_normalizado) * numpy.log(1 - y_pred_normalizado + 1e-8))
        return loss

    def train(self, entradasOriginais: list, nEpocas: int = 100000, isBrekarPorEpocas: bool = True):
        arrPerdas = []
        epoca = 0
        while True:
            epoca += 1
            entradasOriginaisTranspostas = numpy.transpose(entradasOriginais)
            pos_probabilidades_ocultas, pos_estados_ocultas = self.amostraOculta(entradasOriginais)
            pos_associacoes = numpy.dot(entradasOriginaisTranspostas, pos_probabilidades_ocultas)

            neg_probabilidades_visiveis, neg_estados_visiveis = self.amostraVisivel(pos_estados_ocultas)
            neg_probabilidade_ocultas, neg_estados_ocultos = self.amostraOculta(neg_estados_visiveis)

            neg_probabilidades_visiveis_transpostos = numpy.transpose(neg_probabilidades_visiveis)
            neg_associacoes = numpy.dot(neg_probabilidades_visiveis_transpostos, neg_probabilidade_ocultas)


            delta_pesos = (pos_associacoes - neg_associacoes) / len(entradasOriginais)
            delta_bias_oculto = numpy.mean((pos_probabilidades_ocultas - neg_probabilidade_ocultas), axis=0)
            delta_bias_visible = numpy.mean(entradasOriginais - neg_probabilidades_visiveis, axis=0)

            self.pesos += self.txAprendizado * delta_pesos
            self.biasOculto += self.txAprendizado * delta_bias_oculto
            self.biasVisivel += self.txAprendizado * delta_bias_visible

            loss = self.cross_entropia(entradasOriginais, neg_probabilidades_visiveis)

            if(epoca % 100 == 0):
                print(loss)

            if isBrekarPorEpocas:
                if epoca >= nEpocas:
                    break
            else:
                if loss <= 0.0005 or epoca >= nEpocas:
                    break

        return pos_probabilidades_ocultas, pos_estados_ocultas, loss

    def train_saida(self, entradas: list, rotulos: list, nEpocas: int = 10000, isBrekarPorEpocas: bool = True):
        epoca = 0
        saida = []

        while True:
            epoca += 1
            saida = self.sigmoid(numpy.dot(entradas, self.pesos))
            erro = rotulos - saida
            delta_saida = erro * self.derivada_sigmoid(saida)
            self.pesos += self.txAprendizado * numpy.dot(numpy.transpose(entradas), delta_saida)
            loss = self.cross_entropia(rotulos, saida)

            if (epoca % 100 == 0):
                print(loss)

            if isBrekarPorEpocas:
                if epoca >= nEpocas:
                    break
            else:
                if loss <= 0.0005 or epoca >= nEpocas:
                    break

        return saida

    def train_fine_dbn(self, arr_rbms: list[DBN], dados_treinamento: list, rotulos_treinamento: list, nEpocas: int = 500):
        epoca = 0
        saida = []
        isBrekasWhile = False
        while not isBrekasWhile:
            epoca += 1
            arr_camadas_ocultas = [dados_treinamento]
            for rbm in arr_rbms:
                camada_anterior = arr_camadas_ocultas[-1]
                camada_oculta = self.sigmoid(numpy.dot(camada_anterior, rbm.pesos))
                arr_camadas_ocultas.append(camada_oculta)

            saida = self.sigmoid(numpy.dot(arr_camadas_ocultas[-1], self.pesos))

            erro_sadia = saida - rotulos_treinamento
            delta_saida = erro_sadia * self.derivada_sigmoid(saida)
            delta_camada_oculta = numpy.dot(delta_saida, numpy.transpose(self.pesos)) * self.derivada_sigmoid(arr_camadas_ocultas[-1])

            deltas = [delta_camada_oculta]
            for indexRBM in range(len(arr_rbms) - 1, 0, -1):
                delta = numpy.dot(deltas[-1], arr_rbms[indexRBM].pesos.T) * self.derivada_sigmoid(arr_camadas_ocultas[indexRBM])
                deltas.append(delta)

            deltas.reverse()

            for indexRBM in range(len(arr_rbms)):
                camada_anterior = arr_camadas_ocultas[indexRBM]
                delta = deltas[indexRBM]
                arr_rbms[indexRBM].pesos -= self.txAprendizado * numpy.dot(numpy.transpose(camada_anterior), delta)

            self.pesos -= self.txAprendizado * numpy.dot(numpy.transpose(arr_camadas_ocultas[-1]), delta_saida)

            if epoca % 100 == 0:
                loss = self.cross_entropia(rotulos_treinamento, saida)
                print(loss)

            if epoca == nEpocas:
                isBrekasWhile = True

        return saida


    def prever(self, matrizDados: list):
        matrizDadosNormalizada = numpy.asarray(matrizDados)
        h_p, h_s = self.amostraOculta(matrizDadosNormalizada)
        return h_p, h_s


    def prever_dbn_fine(self, arrRBMs: list[DBN], arrDadosPrevisao: list):
        arr_camadas_ocultas = [arrDadosPrevisao]
        for rbm in arrRBMs:
            camada_anterior = arr_camadas_ocultas[-1]
            camada_oculta = self.sigmoid(numpy.dot(camada_anterior, rbm.pesos))
            arr_camadas_ocultas.append(camada_oculta)

        saida = self.sigmoid(numpy.dot(arr_camadas_ocultas[-1], self.pesos))
        return saida


    def treinarDBN(self, dataset: DatasetRNN) -> DatasetRNN:
        arrEntradas = dataset.arr_entradas_treino
        arrSaida = dataset.arr_saidas_esperadas

        print("DBN:", len(arrEntradas), len(arrSaida))

        rbm1 = DBN(len(arrEntradas[0]), 100, 0.1)
        rbm2 = DBN(rbm1.nCamadaOculta, 75, 0.1)
        rbm3 = DBN(rbm2.nCamadaOculta, len(arrSaida[0]), 0.004)

        rbm1.estados_ocultos = rbm1.train(entradasOriginais=arrEntradas, nEpocas=5000, isBrekarPorEpocas=True)[1]
        print("RBM 1 fineshed")
        rbm2.estados_ocultos = rbm2.train(entradasOriginais=rbm1.estados_ocultos, nEpocas=10000, isBrekarPorEpocas=False)[1]
        print("RBM 2 fineshed")
        #rbm3.estados_ocultos = rbm3.train_saida(entradas=rbm2.estados_ocultos, rotulos=arrSaida, nEpocas=10000,
                                                #isBrekarPorEpocas=False)
        print("RBM 3 train saida fineshed")
        #rbm3.estados_ocultos = rbm3.train_fine_dbn(arr_rbms=[rbm1, rbm2], dados_treinamento=arrEntradas,
                                                   #rotulos_treinamento=arrSaida, nEpocas=5000)
        print("RBM 3 fine tunning finished")

        dataset.arr_entradas_treino_DBN = rbm2.estados_ocultos

        #dataset.arr_prevevisao_DBN = rbm3.prever_dbn_fine(arrRBMs=[rbm1, rbm2], arrDadosPrevisao=dataset.arr_prevevisao)
        dataset.arr_prevevisao_DBN = rbm1.prever(matrizDados=dataset.arr_prevevisao)[1]
        dataset.arr_prevevisao_DBN = rbm2.prever(matrizDados=dataset.arr_prevevisao_DBN)[1]

        return dataset

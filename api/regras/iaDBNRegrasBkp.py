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
        #self.pesos = numpy.random.rand(self.nCamadaVisivel, self.nCamadaOculta) * 0.1
        self.biasVisivel = numpy.zeros(nCamadaVisivel)
        self.biasOculto = numpy.zeros(nCamadaOculta)


    def sigmoid(self, x):
        sig = 1 / (1 + numpy.exp(-x))
        return sig


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

    def prever(self, matrizDados: list):
        matrizDadosNormalizada = numpy.asarray(matrizDados)
        h_p, h_s = self.amostraOculta(matrizDadosNormalizada)
        return h_p, h_s

    def treinarDBN(self, dataset: DatasetRNN) -> DatasetRNN:
        arrEntradas = dataset.arr_entradas_treino
        arrSaida = dataset.arr_saidas_esperadas

        print("DBN:", len(arrEntradas), len(arrSaida))

        rbm1 = DBN(len(arrEntradas[0]), int(len(arrEntradas) / 2), 0.1)
        rbm2 = DBN(rbm1.nCamadaOculta, int(rbm1.nCamadaOculta / 2), 0.1)
        rbm3 = DBN(rbm2.nCamadaOculta, len(arrEntradas[0]), 0.1)

        pos_probabilidades_ocultas, pos_estados_ocultas, loss = rbm1.train(entradasOriginais=arrEntradas, nEpocas=3000, isBrekarPorEpocas=True)
        pos_probabilidades_ocultas, pos_estados_ocultas, loss = rbm2.train(entradasOriginais=pos_estados_ocultas, nEpocas=7000, isBrekarPorEpocas=False)
        pos_probabilidades_ocultas, pos_estados_ocultas, loss = rbm3.train(entradasOriginais=pos_estados_ocultas, nEpocas=5000, isBrekarPorEpocas=False)

        dataset.arr_entradas_treino_DBN = pos_estados_ocultas

        #prev_prob, prev_stat = rbm1.prever(matrizDados=dataset.arr_prevevisao)
        #prev_prob, prev_stat = rbm2.prever(matrizDados=prev_stat)
        #prev_prob, prev_stat = rbm3.prever(matrizDados=prev_stat)

        #dataset.arr_prevevisao = prev_stat

        return dataset

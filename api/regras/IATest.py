from __future__ import annotations

import math
import numpy

from matplotlib import pyplot
from api.regras.statisticsRegras import TeamPontuacao

class RNN:
    def normalizar_dataset(self, dataset, max_valor: list = None, min_valor: list = None) -> list|tuple:
        dataset = numpy.asarray(dataset)
        max_valor = numpy.max(dataset, axis=0) if max_valor is None else max_valor
        min_valor = numpy.min(dataset, axis=0) if min_valor is None else min_valor
        dataset_normalizado: list[list] = (dataset - min_valor) / (max_valor - min_valor)

        return dataset_normalizado, max_valor, min_valor


    def __init__(self, nEntrada: int, nOculta: list[list[int, int]], nSaida: int, txAprendizado: float):
        self.nEntrada = nEntrada
        self.nOculta = nOculta
        self.nSaida = nSaida
        self.txAprendizado = txAprendizado

        self.U = numpy.random.uniform(0.1, 0.9, (nOculta, nEntrada))
        self.U = numpy.asarray(self.U, dtype=numpy.float64)
        self.adagrad_U = numpy.zeros_like(self.U)
        self.adagrad_U = numpy.asarray(self.adagrad_U, dtype=numpy.float64)

        self.W = numpy.random.uniform(0.1, 0.9, (nOculta, nOculta))
        self.W = numpy.asarray(self.W, dtype=numpy.float64)
        self.adagrad_W = numpy.zeros_like(self.W)
        self.adagrad_W = numpy.asarray(self.adagrad_W, dtype=numpy.float64)


        self.V = numpy.random.uniform(0.1, 0.9, (nSaida, nOculta))
        self.V = numpy.asarray(self.V, dtype=numpy.float64)
        self.adagrad_V = numpy.zeros_like(self.V)
        self.adagrad_V = numpy.asarray(self.adagrad_V, dtype=numpy.float64)

    def sigmoid(self, x):
        sig = 1 / (1 + numpy.exp(-x))
        return sig

    def derivada_sigmoid(self, x):
        dsig = x * (1 - x)
        return dsig

    def softmax(self, x):
        exp_puntuacao = numpy.exp(x)
        return exp_puntuacao / numpy.sum(exp_puntuacao, axis=0)

    def forward(self, entradas: list, isSaida = False):
        tamanhaEntrada = len(entradas)
        estados_ocultos = numpy.zeros((tamanhaEntrada + 1, self.nOculta), dtype=numpy.float64)
        saidas = numpy.zeros((tamanhaEntrada, self.nSaida), dtype=numpy.float64)

        for t in range(tamanhaEntrada):
            estados_ocultos[t] = self.sigmoid(numpy.dot(self.U, entradas[t]) + numpy.dot(self.W, estados_ocultos[t - 1]))
            if isSaida:
                saidas[t] = self.softmax(numpy.dot(self.V, estados_ocultos[t]))
            else:
                saidas[t] = self.softmax(numpy.dot(self.V, estados_ocultos[t]))

        return estados_ocultos, saidas


    def backward(self, entradas: list, estadosOcultos: list, saidas: list, esperado: list):
        saidas = numpy.asarray(saidas, dtype=numpy.float64)
        entradas = numpy.asarray(entradas, dtype=numpy.float64)
        esperado = numpy.asarray(esperado, dtype=numpy.float64)
        estadosOcultos = numpy.asarray(estadosOcultos, dtype=numpy.float64)

        tamanhoEntrada = len(entradas)
        T = numpy.asarray(entradas, dtype=numpy.float64).shape[0]
        delta_U = numpy.zeros((self.nOculta, self.nEntrada), dtype=numpy.float64)
        delta_W = numpy.zeros((self.nOculta, self.nOculta), dtype=numpy.float64)
        delta_V = numpy.zeros((self.nSaida, self.nOculta), dtype=numpy.float64)

        delta_saidas = saidas - esperado

        for t in range(tamanhoEntrada - 1, -1, -1):
            delta_V += numpy.outer(delta_saidas[t], estadosOcultos[t])
            delta_oculto = numpy.dot(numpy.transpose(self.V), delta_saidas[t]) * (1 - estadosOcultos[t] ** 2)

            for k in range(t, 0, -1):
                delta_W += numpy.outer(delta_oculto, estadosOcultos[k - 1])
                delta_oculto = numpy.dot(numpy.transpose(self.W), delta_oculto) * (1 - estadosOcultos[k - 1] ** 2)

            delta_U += numpy.outer(delta_oculto, entradas[t])

        self.adagrad_W += delta_W ** 2
        self.W -= self.txAprendizado * delta_W / (numpy.sqrt(self.adagrad_W) + 1e-7)

        self.adagrad_U += delta_U ** 2
        self.U -= self.txAprendizado * delta_U / (numpy.sqrt(self.adagrad_U) + 1e-7)

        self.adagrad_V += delta_V ** 2
        self.V -= self.txAprendizado * delta_V / (numpy.sqrt(self.adagrad_V) + 1e-7)

    def treinar(self, entradas, esperados, nepocas, isPlotarGrafico: bool = False):
        arrPerdas = []

        if isPlotarGrafico:
            pyplot.pause(0.1)
            pyplot.plot(arrPerdas)
            pyplot.show(block=False)

        for epoca in range(nepocas):
            estados_ocultos, saida = self.forward(entradas)
            self.backward(entradas, estados_ocultos, saida, esperados)
            loss = numpy.mean(numpy.abs(saida - esperados))

            if isPlotarGrafico:
                arrPerdas.append(loss)
                pyplot.plot(arrPerdas)
                pyplot.pause(0.0001)

            if (epoca + 1) % 2 == 0:
                print("Epoch: ", epoca, ", erro: ", loss)

        if isPlotarGrafico:
            pyplot.close()

        return estados_ocultos, saida

    def prever(self, entrada, isSaida = False):
        estados_ocultos, saida = self.forward(entrada)
        return estados_ocultos, saida

    def normalizarDatasetStatistics(self, arrTeamPontuacao: list[TeamPontuacao]):
        arrChavesKeyResultado = ["resultado", "gols_ocorrido", "gols_marcado", "gols_sofridos"]
        arrChavesKeyDados = []
        arrChaverKeyIgnorar = ["date_fixture"]

        arrAllDadosEntrada = []
        arrAllDadosEsperados = []
        arrAllDadosPrever = []

        for teamPontuacao in arrTeamPontuacao:
            arrDadosEntrada = []
            arrDadosEsperados = []
            arrDadosPrever = []

            for indexModelDados in range(len(teamPontuacao.arr_dataset)):
                modelDados = teamPontuacao.arr_dataset[indexModelDados]
                arrDados = []
                arrResultados = []
                modelDadosDict = modelDados.__dict__

                for key in modelDadosDict.keys():
                    if key in arrChaverKeyIgnorar:
                        continue
                    elif key in arrChavesKeyResultado:
                        #mudar esses -2 para 1 pois é só pra ignorar o mais recente
                        if indexModelDados < len(teamPontuacao.arr_dataset) - 1 :
                            modelDadosPosteriorDict = teamPontuacao.arr_dataset[indexModelDados+1].__dict__
                            arrResultados.append(modelDadosPosteriorDict[key])
                        continue
                    else:
                        arrDados.append(modelDadosDict[key])

                if indexModelDados < len(teamPontuacao.arr_dataset) - 1:
                    arrDados.append(arrResultados)
                    arrDadosEntrada.append(arrDados)

                    arrDadosEsperados.append(arrResultados)
                else:
                    arrDadosEntrada.append(arrDados)

            arrAllDadosEntrada = arrAllDadosEntrada + arrDadosEntrada

        arrAllDados = sorted(arrAllDadosEntrada, key=lambda x: x[1])
        arrDadosEntrada = []
        arrDadosEsperados = []
        arrDadosPrever = []


        for dado in arrAllDados:
            if type(dado[len(dado) -1]) == list:
                arrDadosEsperados.append(dado[-1:][0])
                arrDadosEntrada.append(dado[:-1])
            else:
                arrDadosPrever.append(dado)

        print(arrDadosEntrada)
        print(arrDadosPrever)

        arrDadosEntradaNormalziado, max_valor, min_valor = self.normalizar_dataset(dataset=arrDadosEntrada)
        arrDadosEsperadosNormalizado, max_esp, min_esp = self.normalizar_dataset(dataset=arrDadosEsperados)
        arrDadosPreverNormalizado = self.normalizar_dataset(dataset=arrDadosPrever, max_valor=max_valor, min_valor=min_valor)[0]

        self.__init__(nEntrada=len(arrDadosEntradaNormalziado[0]), nOculta=len(arrDadosEntradaNormalziado[0]) * 2,
                      nSaida=len(arrDadosEsperadosNormalizado[0]), txAprendizado=0.15)

        self.treinar(entradas=arrDadosEntradaNormalziado, esperados=arrDadosEsperadosNormalizado,
                     nepocas=800, isPlotarGrafico=False)

        print(max_esp, min_esp)
        for aaa in arrDadosPreverNormalizado:
            print(self.prever(entrada=[aaa])[1])

 ########## forward que funciona.
    def forward(self, entradas: numpy.ndarray):
        arrSaidasEntradas = []
        arrEstadosOcultos = []

        for index_entrada in range(len(entradas)):
            entrada_t = entradas[index_entrada].reshape(self.nNeuroniosEntrada, 1)
            saida_entrada_t = self.sigmoid(numpy.dot(self.matriz_U[0], entrada_t))

            if len(arrEstadosOcultos) == 0:
                estado_oculto_anterior = numpy.zeros_like(self.matriz_W[0])
                print(estado_oculto_anterior)
                arrEstadosOcultos.append(estado_oculto_anterior)

            for indexCamadaOculta in range(len(self.nNeuroniosCamadaOculta)):
                if indexCamadaOculta == 0:
                    entrada_camada_oculta = saida_entrada_t
                else:
                    entrada_camada_oculta = arrEstadosOcultos[-1]

                dot_W_entrada = numpy.dot(self.matriz_W[indexCamadaOculta], entrada_camada_oculta)

                if indexCamadaOculta == len(self.nNeuroniosCamadaOculta) - 1:
                    dot_U_saida_entrada = numpy.zeros((self.nNeuroniosCamadaOculta[-1], 1))
                else:
                    dot_U_saida_entrada = numpy.dot(self.matriz_U[indexCamadaOculta + 1], arrEstadosOcultos[-1])

                estado_oculto = self.sigmoid(dot_W_entrada + dot_U_saida_entrada)
                arrEstadosOcultos.append(estado_oculto)

            saida = numpy.dot(self.matriz_V, estado_oculto)
            arrSaidasEntradas.append(saida)

        print(arrEstadosOcultos)
        print(arrSaidasEntradas)
        print("####################################")
        return arrEstadosOcultos, arrSaidasEntradas

    def backward(self, entradas, saidas_esperadas, arrSaidas, arrEstadosOcultos):
        arrGrad_U = []
        arrGrad_W = []
        arrGrad_V = []
        arrGrad_B = []

        saidas_esperadas = [numpy.asarray(rotulo).reshape(-1, 1) for rotulo in saidas_esperadas]

        for indexEntrada in reversed(range(len(entradas))):
            erro = arrSaidas[indexEntrada] - saidas_esperadas[indexEntrada]
            estado_oculto = arrEstadosOcultos[indexEntrada][-1]
            delta_saida = (erro * self.derivada_sigmoid(arrSaidas[indexEntrada])).tolist()
            grad_V = numpy.dot(delta_saida, estado_oculto.T)
            grad_B = delta_saida

            delta_oculto = numpy.dot(self.matriz_V.T, delta_saida) * self.derivada_tanh(estado_oculto)
            grad_W.append(numpy.dot(delta_oculto, arrEstadosOcultos[indexEntrada][-2].T))
            grad_B.append(delta_oculto)

            for indexCamadaOculta in range(len(self.nNeuroniosCamadaOculta) - 2, -1, -1):
                delta_oculto = numpy.dot(self.matriz_W[indexCamadaOculta + 1].T, delta_oculto) * self.derivada_tanh(
                    arrEstadosOcultos[indexEntrada][indexCamadaOculta + 1])
                grad_W.append(numpy.dot(delta_oculto, arrEstadosOcultos[indexEntrada][indexCamadaOculta].T))
                grad_B.append(delta_oculto)

            grad_W.reverse()
            grad_B.reverse()

            grad_U.append(numpy.dot(delta_oculto, entradas[indexEntrada].reshape(-1, 1).T))

            for indexCamadaOculta in range(1, len(self.nNeuroniosCamadaOculta)):
                delta_oculto = numpy.dot(self.matriz_W[indexCamadaOculta].T, delta_oculto) * self.derivada_tanh(
                    arrEstadosOcultos[indexEntrada][indexCamadaOculta])
                grad_U.append(numpy.dot(delta_oculto, arrEstadosOcultos[indexEntrada][indexCamadaOculta - 1].T))

            arrGrad_U.append(grad_U)
            arrGrad_W.append(grad_W)
            arrGrad_V.append(grad_V)
            arrGrad_B.append(grad_B)

        grad_U_finais = []
        grad_W_finais = numpy.zeros((len(entradas), len(self.nNeuroniosCamadaOculta)))
        grad_B_finais = []
        for i in range(len(self.nNeuroniosCamadaOculta)):
            grad_U_finais.append(numpy.zeros_like(self.matriz_U[i]))
            grad_B_finais.append(numpy.zeros_like(self.matriz_B[i]))

        grad_V_final = numpy.zeros_like(self.matriz_V)
        grad_B_final = numpy.zeros_like(self.matriz_B[-1])

        for i in range(len(self.nNeuroniosCamadaOculta)):
            for indexEntrada in range(len(entradas) - 1, 0, -1):
                grad_W_finais[i] += arrGrad_W[indexEntrada][i]
                grad_U_finais[i] += arrGrad_U[indexEntrada][i]
                grad_B_finais[i] += arrGrad_B[indexEntrada][i]

                self.matriz_W[i] -= self.taxaAprendizado * grad_W_finais[i]
                self.matriz_U[i] -= self.taxaAprendizado * grad_U_finais[i]
                self.matriz_B[i] -= self.taxaAprendizado * grad_B_finais[i]

        grad_V_final += arrGrad_V[len(arrGrad_V) - 1]
        grad_B_final += arrGrad_B[len(arrGrad_B) - 1][-1]
        self.matriz_V -= self.taxaAprendizado * grad_V_final
        self.matriz_B[-1][-1] -= self.taxaAprendizado * grad_B_final

        return grad_U_final, grad_W_final, grad_V_final, grad_B_final
from __future__ import annotations

import math
import numpy

from copy import deepcopy
from matplotlib import pyplot
from api.regras.iaUteisRegras import IAUteisRegras


class ModelDataRNN:
    def __init__(self, arrEntradas: list[list], arrRotulos: list[list[list]],
                 arrDadosPrever: list[list[list]] = None, arrNameFuncAtivacaoCadaOculta: list[str] = [],
                 arrNameFuncAtivacaoCadaSaida: list[str] = []):

        if len(arrRotulos[0][0]) == 0:
            raise "reveja os rótulos nao é uma list[list[list]]"

        self.iaRegras = IAUteisRegras()
        self.n_epocas: int = 25000
        self.taxa_aprendizado: float = 0.1
        self.taxa_regularizacao_l2: float = 0.001

        self.arr_n_camada_oculta: list[int] = [len(arrEntradas[0])]
        self.nNeuroniosEntrada: int = len(arrEntradas[0])
        self.arrCamadasSaida: list[int] = [len(saida) for saida in arrRotulos[0]]

        self.arr_entradas: list[list] = arrEntradas
        # Lista de dados com os dados divididos em camadas lista rotulos com lista de camada
        self.arr_rotulos: list[list[list]] = arrRotulos
        self.arr_dados_prever: list[list[list]] = arrDadosPrever
        self.arrFuncAtivacaoCadaSaida: list[list[any, any]] = []
        self.arrFuncAtivacaoCamadaOculta: list[list[any, any]] = []

        if len(arrNameFuncAtivacaoCadaSaida) == 0:
            for camadaSaida in self.arr_rotulos[0]:
                arrNameFuncAtivacaoCadaSaida.append("sigmoid")

        if len(arrNameFuncAtivacaoCadaSaida) == 1 and len(self.arrCamadasSaida) >= 2:
            for indexCamadaSaida in range(len(self.arrCamadasSaida)):
                if indexCamadaSaida >= 1:
                    arrNameFuncAtivacaoCadaSaida.append(arrNameFuncAtivacaoCadaSaida[0])

        for name in arrNameFuncAtivacaoCadaSaida:
            arrFunctActivDeriv = []
            if name == "softmax":
                arrFunctActivDeriv = [self.iaRegras.softmax, self.iaRegras.derivada_softmax]
                self.arrFuncAtivacaoCadaSaida.append(arrFunctActivDeriv)
            elif name == "sigmoid":
                arrFunctActivDeriv = [self.iaRegras.sigmoid, self.iaRegras.derivada_sigmoid]
                self.arrFuncAtivacaoCadaSaida.append(arrFunctActivDeriv)
            else:
                raise "Functions de ativação nao edfinidos"

        if len(arrNameFuncAtivacaoCadaOculta) == 0:
            for camadaSaida in self.arr_n_camada_oculta:
                arrNameFuncAtivacaoCadaOculta.append("sigmoid")

        if len(arrNameFuncAtivacaoCadaOculta) == 1 and len(self.arr_n_camada_oculta) >= 2:
            for indexCamada in range(len(self.arr_n_camada_oculta)):
                if indexCamada >= 1:
                    arrNameFuncAtivacaoCadaOculta.append(arrNameFuncAtivacaoCadaOculta[0])

        for name in arrNameFuncAtivacaoCadaOculta:
            arrFunctActivDeriv = []
            if name == "tanh":
                arrFunctActivDeriv = [self.iaRegras.tanh, self.iaRegras.derivada_tanh]
                self.arrFuncAtivacaoCamadaOculta.append(arrFunctActivDeriv)
            elif name == "sigmoid":
                arrFunctActivDeriv = [self.iaRegras.sigmoid, self.iaRegras.derivada_sigmoid]
                self.arrFuncAtivacaoCamadaOculta.append(arrFunctActivDeriv)
            else:
                raise "Functions de ativação nao edfinidos"

        self.arrNameFuncCamadaSaida = arrNameFuncAtivacaoCadaSaida
        self.arrNameFuncCamadaOculta = arrNameFuncAtivacaoCadaOculta

class RNN:
    def __init__(self, modelDataRNN: ModelDataRNN, isNovosPesos: bool = True):
        self.iaRegras = IAUteisRegras()
        self.modelDataRNN = modelDataRNN
        self.nNeuroniosEntrada = modelDataRNN.nNeuroniosEntrada
        self.arrNCamadasOcultas = modelDataRNN.arr_n_camada_oculta
        self.arrCamadasSaida = modelDataRNN.arrCamadasSaida
        self.txAprendizado = modelDataRNN.taxa_aprendizado
        self.taxa_regularizacao = modelDataRNN.taxa_regularizacao_l2
        self.nEpocas = modelDataRNN.n_epocas
        self.media_entropy = 0
        self.media_accuracy = 0

        if not isNovosPesos:
            return

        self.matriz_U: list = []
        self.matriz_W: list = []
        self.matriz_V: list = []

        self.matriz_Ub: list = []
        self.matriz_Wb: list = []
        self.matriz_Vb: list = []

        self.matriz_adagrad_U: list = []
        self.matriz_adagrad_W: list = []
        self.matriz_adagrad_V: list = []

        self.matriz_adagrad_Ub: list = []
        self.matriz_adagrad_Wb: list = []
        self.matriz_adagrad_Vb: list = []

        for indexnNeuroniosOcultos in range(len(self.arrNCamadasOcultas)):
            nNeuroniosCmdAtual = self.arrNCamadasOcultas[indexnNeuroniosOcultos]
            nNeuroniosCmdAnterior = self.nNeuroniosEntrada if indexnNeuroniosOcultos == 0 else \
                self.arrNCamadasOcultas[indexnNeuroniosOcultos - 1]

            W = self.inicalizarPesosXavier(nNeuroniosCmdAtual, (nNeuroniosCmdAtual, nNeuroniosCmdAnterior),
                                           isScalaMenorZero=
                                           self.modelDataRNN.arrNameFuncCamadaOculta[indexnNeuroniosOcultos] == "tanh")

            Wb = self.inicalizarPesosXavier(nNeuroniosCmdAtual, (nNeuroniosCmdAtual, nNeuroniosCmdAnterior),
                                            isReturnMatriz=False,
                                            isScalaMenorZero=
                                            self.modelDataRNN.arrNameFuncCamadaOculta[indexnNeuroniosOcultos] == "tanh")

            U = self.inicalizarPesosXavier(nNeuroniosCmdAtual, (nNeuroniosCmdAtual, nNeuroniosCmdAtual),
                                           isScalaMenorZero=
                                           self.modelDataRNN.arrNameFuncCamadaOculta[indexnNeuroniosOcultos] == "tanh")

            Ub = self.inicalizarPesosXavier(nNeuroniosCmdAtual, (nNeuroniosCmdAtual, nNeuroniosCmdAnterior),
                                            isReturnMatriz=False,
                                            isScalaMenorZero=
                                            self.modelDataRNN.arrNameFuncCamadaOculta[indexnNeuroniosOcultos] == "tanh")

            self.matriz_adagrad_W.append(numpy.zeros_like(W))
            self.matriz_adagrad_U.append(numpy.zeros_like(U))
            self.matriz_adagrad_Wb.append(numpy.zeros_like(Wb))
            self.matriz_adagrad_Ub.append(numpy.zeros_like(Ub))

            self.matriz_W.append(W)
            self.matriz_U.append(U)
            self.matriz_Wb.append(Wb)
            self.matriz_Ub.append(Ub)

        for nNeuroniosSaida in range(len(self.arrCamadasSaida)):
            V = self.inicalizarPesosXavier((self.arrCamadasSaida[nNeuroniosSaida]),
                                           (self.arrCamadasSaida[nNeuroniosSaida], self.arrNCamadasOcultas[-1]),
                                           isScalaMenorZero=self.modelDataRNN.arrNameFuncCamadaSaida[nNeuroniosSaida] == "tanh")

            Vb = self.inicalizarPesosXavier((self.arrCamadasSaida[nNeuroniosSaida]),
                                            (self.arrCamadasSaida[nNeuroniosSaida], self.arrNCamadasOcultas[-1]),
                                            isReturnMatriz=False,
                                            isScalaMenorZero=self.modelDataRNN.arrNameFuncCamadaSaida[nNeuroniosSaida] == "tanh")

            self.matriz_adagrad_V.append(numpy.zeros_like(V))
            self.matriz_adagrad_Vb.append(numpy.zeros_like(Vb))

            self.matriz_V.append(V)
            self.matriz_Vb.append(Vb)

    def inicalizarPesosXavier(self, nItens: int, tupleDim: tuple, isScalaMenorZero: bool = False,
                              isReturnMatriz: bool = True) -> list:
        initScale = -numpy.sqrt(2.0 / int(nItens)) if isScalaMenorZero else 0
        endScale = numpy.sqrt(2.0 / int(nItens))

        if isReturnMatriz:
            arrXavier = numpy.random.uniform(initScale, endScale, tupleDim)
        else:
            arrXavier = numpy.random.uniform(initScale, endScale, tupleDim[0])

        return arrXavier

    def forward(self, entradas: list, estado_oculto_anterior: list = None):
        arrSaidas = []
        arrEstadosOcultos = []

        for indexEntrada in range(len(entradas)):
            entrada = entradas[indexEntrada]
            entrada_t = numpy.transpose([entrada])
            arrEstadosOcultos.append([])

            for indexCamadaOculta in range(len(self.arrNCamadasOcultas)):
                funcAtivacaoOculta = self.modelDataRNN.arrFuncAtivacaoCamadaOculta[indexCamadaOculta][0]

                if indexEntrada == 0:
                    estado_oculto_t = numpy.transpose([numpy.zeros(self.arrNCamadasOcultas[indexCamadaOculta])]) \
                        if estado_oculto_anterior is None else estado_oculto_anterior[indexEntrada][indexCamadaOculta]
                else:
                    estado_oculto_t = arrEstadosOcultos[indexEntrada - 1][indexCamadaOculta]

                if indexCamadaOculta == 0:
                    entrada_t = entrada_t
                else:
                    entrada_t = arrEstadosOcultos[indexEntrada][indexCamadaOculta - 1]

                dot_W = numpy.dot(self.matriz_W[indexCamadaOculta], entrada_t)
                dot_U = numpy.dot(self.matriz_U[indexCamadaOculta], estado_oculto_t)

                sum_dot_U_W = dot_W + dot_U
                estado_oculto_n = funcAtivacaoOculta(sum_dot_U_W)

                arrEstadosOcultos[indexEntrada].append(estado_oculto_n)

            arrSaidas_t = []

            for indexCamadaSaida in range(len(self.arrCamadasSaida)):
                funcAtivacaoSaida = self.modelDataRNN.arrFuncAtivacaoCadaSaida[indexCamadaSaida][0]
                dot_saida = numpy.dot(self.matriz_V[indexCamadaSaida], arrEstadosOcultos[indexEntrada][-1])
                saida_camada_t = funcAtivacaoSaida(dot_saida)
                arrSaidas_t.append(saida_camada_t)

            arrSaidas.append(arrSaidas_t)

        return arrSaidas, arrEstadosOcultos

    def backward(self, entradas: list, esperado: list, saidas: list, estadosOcultos: list):
        lambda_reg = 0.01
        delta_W = [numpy.zeros_like(self.matriz_W[i]) for i in range(len(self.arrNCamadasOcultas))]
        delta_U = [numpy.zeros_like(self.matriz_U[i]) for i in range(len(self.arrNCamadasOcultas))]
        delta_V = [numpy.zeros_like(self.matriz_V[i]) for i in range(len(self.arrCamadasSaida))]

        delta_Wb = [numpy.zeros_like(self.matriz_Wb[i]) for i in range(len(self.arrNCamadasOcultas))]
        delta_Ub = [numpy.zeros_like(self.matriz_Ub[i]) for i in range(len(self.arrNCamadasOcultas))]
        delta_Vb = [numpy.zeros_like(self.matriz_Vb[i]) for i in range(len(self.arrCamadasSaida))]

        for index_entrada_t in reversed(range(len(entradas))):
            delta_oculto = [numpy.zeros((i, 1)) for i in self.arrNCamadasOcultas]
            delta_saida = [numpy.zeros((i, self.arrNCamadasOcultas[-1])) for i in self.arrCamadasSaida]
            arr_deltas_ocultos = []
            estados_ocultos_t = estadosOcultos[index_entrada_t]
            entrada_t = numpy.transpose([entradas[index_entrada_t]])
            rotulo_t = esperado[index_entrada_t]
            saida_t = saidas[index_entrada_t]
            last_estado_oculto = estadosOcultos[index_entrada_t][-1]
            funcDerivadaLastOculta = self.modelDataRNN.arrFuncAtivacaoCamadaOculta[-1][1]

            for index_camada_saida in reversed(range(len(self.arrCamadasSaida))):
                funcDerivadaSaida = self.modelDataRNN.arrFuncAtivacaoCadaSaida[index_camada_saida][1]
                erro_t = (saida_t[index_camada_saida] - numpy.transpose([rotulo_t[index_camada_saida]])) * funcDerivadaSaida(saida_t[index_camada_saida])

                delta_saida[index_camada_saida] = numpy.clip((numpy.dot(erro_t, numpy.transpose(last_estado_oculto)) +
                                                              delta_saida[index_camada_saida]), -2, 2)

                delta_V[index_camada_saida] += delta_saida[index_camada_saida]
                # delta_Vb[index_camada_saida] += erro_t
                delta_oculto[-1] += numpy.dot(self.matriz_V[index_camada_saida].T, erro_t) * funcDerivadaLastOculta(last_estado_oculto)

                # self.matriz_adagrad_Vb[index_camada_saida] += delta_Vb[index_camada_saida] ** 2
                # self.matriz_Vb[index_camada_saida] -= self.txAprendizado * delta_Vb[index_camada_saida] / \
                # (numpy.sqrt(self.matriz_adagrad_Vb[index_camada_saida]) + 1e-9)

                delta_V_regularized = delta_V[index_camada_saida] + self.taxa_regularizacao * self.matriz_V[index_camada_saida]
                self.matriz_adagrad_V[index_camada_saida] += delta_V[index_camada_saida] ** 2
                self.matriz_V[index_camada_saida] -= (self.txAprendizado * delta_V_regularized) / \
                                                     (numpy.sqrt(self.matriz_adagrad_V[index_camada_saida]) + 1e-7)

            for index_camada_oculta in reversed(range(len(self.arrNCamadasOcultas))):
                funcDerivadaOculta = self.modelDataRNN.arrFuncAtivacaoCamadaOculta[index_camada_oculta][1]

                estado_oculto_camada_anterior = numpy.transpose(entrada_t) if index_camada_oculta == 0 else \
                    numpy.transpose(estados_ocultos_t[index_camada_oculta - 1])

                estado_oculto_camada_atual = numpy.transpose(estados_ocultos_t[index_camada_oculta])
                estado_oculto_camada_atual_t_anterior = numpy.transpose(estadosOcultos[index_entrada_t - 1][index_camada_oculta]) \
                    if index_entrada_t >= 1 else numpy.transpose(numpy.zeros_like(estados_ocultos_t[index_camada_oculta]))

                delta_W[index_camada_oculta] = (
                    numpy.clip(numpy.dot(delta_oculto[index_camada_oculta], estado_oculto_camada_anterior) +
                               delta_W[index_camada_oculta], -2, 2))
                delta_U[index_camada_oculta] = (
                    numpy.clip(numpy.dot(delta_oculto[index_camada_oculta], estado_oculto_camada_atual_t_anterior) +
                               delta_U[index_camada_oculta], -2, 2))

                if index_camada_oculta >= 1:
                    dot_delta_oculto = numpy.dot(self.matriz_W[index_camada_oculta].T, delta_oculto[index_camada_oculta])
                    deriv_delta_oculto = funcDerivadaOculta(numpy.transpose(estado_oculto_camada_anterior))
                    delta_oculto[index_camada_oculta - 1] = dot_delta_oculto * deriv_delta_oculto

                delta_W_regularized = delta_W[index_camada_oculta] + self.taxa_regularizacao * self.matriz_W[index_camada_oculta]
                self.matriz_adagrad_W[index_camada_oculta] += self.matriz_W[index_camada_oculta] ** 2
                self.matriz_W[index_camada_oculta] -= (self.txAprendizado * delta_W_regularized) / \
                                                      (numpy.sqrt(self.matriz_adagrad_W[index_camada_oculta]) + 1e-7)

                delta_Wb_regularized = delta_Wb[index_camada_oculta] + self.taxa_regularizacao * self.matriz_Wb[index_camada_oculta]
                self.matriz_adagrad_Wb[index_camada_oculta] += self.matriz_Wb[index_camada_oculta] ** 2
                self.matriz_Wb[index_camada_oculta] -= (self.txAprendizado * delta_Wb_regularized) / \
                                                      (numpy.sqrt(self.matriz_adagrad_Wb[index_camada_oculta]) + 1e-7)

                delta_U_regularized = delta_U[index_camada_oculta] + self.taxa_regularizacao * self.matriz_U[index_camada_oculta]
                self.matriz_adagrad_U[index_camada_oculta] += delta_U_regularized ** 2
                self.matriz_U[index_camada_oculta] -= (self.txAprendizado * delta_U_regularized) / \
                                                      (numpy.sqrt(self.matriz_adagrad_U[index_camada_oculta]) + 1e-7)

                delta_Ub_regularized = delta_Ub[index_camada_oculta] + self.taxa_regularizacao * self.matriz_Ub[index_camada_oculta]
                self.matriz_adagrad_Ub[index_camada_oculta] += self.matriz_Ub[index_camada_oculta] ** 2
                self.matriz_Ub[index_camada_oculta] -= (self.txAprendizado * delta_Ub_regularized) / \
                                                      (numpy.sqrt(self.matriz_adagrad_Ub[index_camada_oculta]) + 1e-7)

    def calcular_erro(self, rotulos: list[list[list]], arrPrevisoes: list[list[list[list]]], isPrintar: bool = True) -> \
            tuple[list, list]:
        arrEntropy, arrAcurracy = [], []

        for index_previsoes in range(len(arrPrevisoes)):
            previsoes = arrPrevisoes[index_previsoes]
            entropy_for_camadas = [0 for i in self.arrCamadasSaida]
            accuracy_for_camada = [0 for i in self.arrCamadasSaida]
            sum_values_accuracy = [0 for i in self.arrCamadasSaida]

            for indexClasseSaida in range(len(self.arrCamadasSaida)):
                for indexDado in range(len(previsoes)):
                    saidaCamada = numpy.reshape(previsoes[indexDado][indexClasseSaida], (1, -1))[0]
                    rotulo = rotulos[indexDado][indexClasseSaida]

                    dotEntropy = rotulo * numpy.log(saidaCamada + 1e-9)
                    entropy_for_camadas[indexClasseSaida] += -numpy.sum(dotEntropy, axis=0)

                    if len(saidaCamada) >= 2 and self.modelDataRNN.arrNameFuncCamadaSaida[indexClasseSaida] == "softmax":
                        maxArgSaida = numpy.argmax(saidaCamada)
                        maxArgRotulo = numpy.argmax(rotulo)
                        if maxArgRotulo == maxArgSaida:
                            sum_values_accuracy[indexClasseSaida] += 1
                    else:
                        sumAccuracy = []
                        for indexDadoSaida in range(len(saidaCamada)):
                            throbleshot = 0.5
                            y = int(saidaCamada[indexDadoSaida] >= throbleshot)
                            if y == rotulo[indexDadoSaida]:
                                sumAccuracy.append(1)
                            else:
                                sumAccuracy.append(0)

                        sums = sum(sumAccuracy) / len(sumAccuracy) if sum(sumAccuracy) > 0 else 0

                        if sums == 1:
                            sum_values_accuracy[indexClasseSaida] += 1

            msgEntropy = "Entropy for camada "
            msgAcurracy = "Acurracy for camada "
            for indexCamadaSaida in range(len(self.arrCamadasSaida)):
                accuracy_for_camada[indexCamadaSaida] = sum_values_accuracy[indexCamadaSaida] / len(rotulos)

                if isPrintar:
                    # msgEntropy += f" [{indexCamadaSaida}: {entropy_for_camadas[indexCamadaSaida]:.5f}], "
                    # msgAcurracy += f" [{indexCamadaSaida}: {accuracy_for_camada[indexCamadaSaida]:.5f}], "
                    pass

            if isPrintar:
                # print(msgEntropy, msgAcurracy)
                # print("######################")
                pass

            arrEntropy.append(entropy_for_camadas)
            arrAcurracy.append(accuracy_for_camada)

        mediaEntropy = numpy.sum(arrEntropy, axis=0) / len(arrEntropy)
        mediaAcurracy = numpy.sum(arrAcurracy, axis=0) / len(arrAcurracy)
        msgMediaEntropy = " Media entropy for camada"
        msgMediaAcurracy = "Media acuracy for camada"

        for indexCamadaSaida in range(len(self.arrCamadasSaida)):
            accuracy_for_camada[indexCamadaSaida] = sum_values_accuracy[indexCamadaSaida] / len(rotulos)
            msgMediaEntropy += f" [{indexCamadaSaida}: {mediaEntropy[indexCamadaSaida]:.5f}], "
            msgMediaAcurracy += f" [{indexCamadaSaida}: {mediaAcurracy[indexCamadaSaida]:.5f}], "

        if isPrintar:
            print(msgMediaEntropy, "\n", msgMediaAcurracy)
            print("##########")

        return mediaEntropy, mediaAcurracy

    def treinar(self, isBrekarPorEpocas: bool = True, isAtualizarPesos: bool = True, qtdeDadoValidar: int = 2,
                n_folds: int = 5, isRecursiva: bool = False, isForcarTreino: bool = False) -> tuple[list, list, list] | bool:

        nEpoca = 0
        nEpocaValidacao = 0
        isBrekarWhile = False
        arrEntradas = deepcopy(self.modelDataRNN.arr_entradas)
        arrRotulos = deepcopy(self.modelDataRNN.arr_rotulos)

        arrArrSaidas = []
        arrEntradas_t = arrEntradas
        arrRotulos_t = arrRotulos
        media_entropy, media_accuracy = [0], [0]
        if qtdeDadoValidar >= 1:
            arrEntradas_v = arrEntradas[-qtdeDadoValidar:]
            arrRotulos_v = arrRotulos[-qtdeDadoValidar:]

            for i in range(qtdeDadoValidar):
                arrEntradas_t.pop()
                arrRotulos_t.pop()
        else:
            arrEntradas_v = arrEntradas[-int(len(arrEntradas_t)):]
            arrRotulos_v = arrRotulos[-int(len(arrRotulos_t)):]

        previsao = []
        while not isBrekarWhile:
            nEpoca += 1
            previsoes, estados_ocultos = self.forward(entradas=arrEntradas_t)
            arrArrSaidas.append(previsoes)

            if nEpoca % 100 == 0:
                print(nEpoca, "de", self.nEpocas, ", txAprendizado: ", self.txAprendizado,
                      ", L2: ", self.taxa_regularizacao, ", qtdeDados: ", len(arrEntradas),
                      ", camadas: ", self.arrNCamadasOcultas, ", nEntrada: ", len(arrEntradas[0]))

            media_entropy, media_accuracy = self.calcular_erro(rotulos=arrRotulos_t, arrPrevisoes=arrArrSaidas,
                                                               isPrintar=nEpoca % 100 == 0)
            arrArrSaidas = []

            if sum(media_accuracy) / len(media_accuracy) >= 0.95:
                isAcuraciaBoa = True
            else:
                isAcuraciaBoa = False

            if sum(media_accuracy) / len(media_accuracy) >= 1 and not isRecursiva and nEpoca >= int(self.nEpocas):
                print("Não foi possível encontrar um bom resultado. Chegaram a 1")

            if isAcuraciaBoa:
                previsoes_v, estados_ocultos_v = (
                    self.forward(entradas=arrEntradas_v,
                                 estado_oculto_anterior=[estados_ocultos[-1]] if qtdeDadoValidar >= 1 else None))

                if nEpoca % 10 == 0:
                    print("############ Validação #############")

                media_entropy_v, media_accuracy_v = self.calcular_erro(rotulos=arrRotulos_v, arrPrevisoes=[previsoes_v],
                                                                       isPrintar=nEpoca % 10 == 0)

                if ((sum(media_accuracy_v) / len(media_accuracy_v) >= 0.5 and not isForcarTreino) or
                        (isForcarTreino and (sum(media_accuracy_v) / len(media_accuracy_v) >= 1))):
                    if self.modelDataRNN.arr_dados_prever is not None:

                        previsao = self.forward(entradas=self.modelDataRNN.arr_dados_prever,
                                                estado_oculto_anterior=[estados_ocultos_v[-1]])[0]

                        if nEpoca == self.nEpocas:
                            for prev in previsao:
                                print("Previsão: ", [a for a in prev])
                        break
                elif sum(media_accuracy) == 1:
                    return False

            if isBrekarPorEpocas and nEpoca == self.nEpocas:
                print("Não foi possível encontrar um bom resultado. 3")
                if not isAtualizarPesos:
                    break
                else:
                    return False

            if isAtualizarPesos:
                self.backward(entradas=arrEntradas_t, esperado=arrRotulos_t, estadosOcultos=estados_ocultos,
                              saidas=previsoes)

        self.media_entropy = media_entropy
        self.media_accuracy = media_accuracy

        return media_entropy, media_accuracy, previsao

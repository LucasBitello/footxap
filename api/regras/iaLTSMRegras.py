from __future__ import annotations
from scipy.special import expit
import numpy as numpy
import random as random
from copy import deepcopy
from api.regras.iaRegras import IARegras

class ModelDataLTSM:
    def __init__(self, arrEntradas: list[list], arrRotulos: list[list[list]], arrRotulosOriginais: list[list[list]],
                 arrDadosPrever: list[list[list]] = None, arrNameFuncAtivacaoCadaSaida: list[str] = []):

        if len(arrRotulos[0][0]) == 0:
            raise "reveja os rótulos nao é uma list[list[list]]"

        self.iaRegras = IARegras()
        self.n_epocas: int = 1500
        self.taxa_aprendizado: float = 0.05
        self.taxa_regularização_l2: float = 0.001

        self.arr_n_camada_oculta: list[int] = [0]
        self.nNeuroniosEntrada: int = len(arrEntradas[0])
        self.arrCamadasSaida: list[int] = [len(saida) for saida in arrRotulos[0]]

        self.arr_entradas: list[list] = arrEntradas
        # Lista de dados com os dados divididos em camadas lista rotulos com lista de camada
        self.arr_rotulos: list[list[list]] = arrRotulos
        self.arr_rotulos_originais: list[list[list]] = arrRotulosOriginais
        self.arr_dados_prever: list[list[list]] = arrDadosPrever
        self.arrFuncAtivacaoCadaSaida: list[list[any, any]] = []

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


class LSTM:
    def __init__(self,  modelDataLTSM: ModelDataLTSM):
        self.modelDataLTSM = modelDataLTSM
        self.nEpocas = modelDataLTSM.n_epocas
        self.txAprendizado = modelDataLTSM.taxa_aprendizado
        self.arrCamadaSaida = modelDataLTSM.arrCamadasSaida
        self.nNeuroniosEntrada = modelDataLTSM.nNeuroniosEntrada
        self.arrNEstadosOcultos = modelDataLTSM.arr_n_camada_oculta
        self.lambdaRegAdagrad = modelDataLTSM.taxa_regularização_l2


        self.Wf, self.Wi, self.Wc, self.Wo, self.Wv = [], [], [], [], []
        self.AdaWf, self.AdaWi, self.AdaWc, self.AdaWo, self.AdaWv = [], [], [], [], []
        self.bf, self.bi, self.bc, self.bo, self.bv = [], [], [], [], []

        self.saidas_rede: list = []
        self.previsoes_rede: list = []
        self.celulas_rede: list = [[] for _ in range(len(self.arrNEstadosOcultos))]
        self.estados_ocultos_rede: list = [[] for _ in range(len(self.arrNEstadosOcultos))]

        self.all_f = [[] for _ in range(len(self.arrNEstadosOcultos))]
        self.all_i = [[] for _ in range(len(self.arrNEstadosOcultos))]
        self.all_o = [[] for _ in range(len(self.arrNEstadosOcultos))]
        self.all_c = [[] for _ in range(len(self.arrNEstadosOcultos))]

        arrNEstadosOcultosAnterior = []
        for indexNNeuronios in range(len(self.arrNEstadosOcultos)):
            if indexNNeuronios == 0:
                tamanho_estado_oculto = self.arrNEstadosOcultos[0]
                tamanho_estado_anterior = self.arrNEstadosOcultos[indexNNeuronios] + self.nNeuroniosEntrada
                arrNEstadosOcultosAnterior.append(tamanho_estado_oculto)
            else:
                tamanho_estado_oculto = self.arrNEstadosOcultos[indexNNeuronios]
                tamanho_estado_anterior = arrNEstadosOcultosAnterior[indexNNeuronios - 1]
                arrNEstadosOcultosAnterior.append(tamanho_estado_oculto)

            somaNeuroniosSaida = sum(self.arrCamadaSaida)
            if indexNNeuronios < len(self.arrNEstadosOcultos) - 1:
                somaNeuroniosSaida = self.arrNEstadosOcultos[indexNNeuronios + 1]
            else:
                somaNeuroniosSaida = sum(self.arrCamadaSaida)

            self.Wf.append(self.inicalizarPesosXavier(somaNeuroniosSaida, (tamanho_estado_oculto, tamanho_estado_anterior)))
            self.Wi.append(self.inicalizarPesosXavier(somaNeuroniosSaida, (tamanho_estado_oculto, tamanho_estado_anterior)))
            self.Wc.append(self.inicalizarPesosXavier(somaNeuroniosSaida, (tamanho_estado_oculto, tamanho_estado_anterior)))
            self.Wo.append(self.inicalizarPesosXavier(somaNeuroniosSaida, (tamanho_estado_oculto, tamanho_estado_anterior)))

            self.AdaWf.append(numpy.zeros_like(self.Wf[-1]))
            self.AdaWi.append(numpy.zeros_like(self.Wi[-1]))
            self.AdaWc.append(numpy.zeros_like(self.Wc[-1]))
            self.AdaWo.append(numpy.zeros_like(self.Wo[-1]))

            self.bf.append(numpy.zeros((tamanho_estado_oculto, 1)))
            self.bi.append(numpy.zeros((tamanho_estado_oculto, 1)))
            self.bc.append(numpy.zeros((tamanho_estado_oculto, 1)))
            self.bo.append(numpy.zeros((tamanho_estado_oculto, 1)))

        for nNeuroniosSaida in self.arrCamadaSaida:
            self.Wv.append(self.inicalizarPesosXavier(somaNeuroniosSaida, (nNeuroniosSaida, self.arrNEstadosOcultos[-1])))
            self.AdaWv.append(numpy.zeros_like(self.Wv[-1]))
            self.bv.append(numpy.zeros((nNeuroniosSaida, 1)))

    def inicalizarPesosXavier(self, nItens: int, tupleDim: tuple) -> list:
        arrXavier = numpy.random.uniform(-numpy.sqrt(2 / nItens), numpy.sqrt(2 / nItens), tupleDim)
        return arrXavier

    def init_new_epoca(self):
        self.saidas_rede: list = []
        self.celulas_rede: list = [[] for _ in range(len(self.arrNEstadosOcultos))]
        self.estados_ocultos_rede: list = [[] for _ in range(len(self.arrNEstadosOcultos))]

        self.all_f = [[] for _ in range(len(self.arrNEstadosOcultos))]
        self.all_i = [[] for _ in range(len(self.arrNEstadosOcultos))]
        self.all_o = [[] for _ in range(len(self.arrNEstadosOcultos))]
        self.all_c = [[] for _ in range(len(self.arrNEstadosOcultos))]

    def forward(self, entradas: list, isPrevisao: bool = False) -> list[list[list]]:
        self.init_new_epoca()
        celulas_prev = [numpy.zeros((i, 1)) for i in self.arrNEstadosOcultos]
        estados_ocultos_prev = [numpy.zeros((i, 1)) for i in self.arrNEstadosOcultos]

        for index_entrada_t in range(len(entradas)):
            entrada_t = numpy.reshape(entradas[index_entrada_t], (-1, 1))

            for index_camada_oculta in range(len(self.arrNEstadosOcultos)):
                if index_camada_oculta == 0:
                    estado_oculto_anterior = numpy.concatenate((estados_ocultos_prev[index_camada_oculta], entrada_t), axis=0)
                else:
                    estado_oculto_anterior = estados_ocultos_prev[index_camada_oculta - 1]

                Wf_dot = numpy.dot(self.Wf[index_camada_oculta], estado_oculto_anterior)
                f = self.modelDataLTSM.iaRegras.sigmoid(Wf_dot + self.bf[index_camada_oculta] + 1e-7)
                self.all_f[index_camada_oculta].append(f)

                Wi_dot = numpy.dot(self.Wi[index_camada_oculta], estado_oculto_anterior)
                i = self.modelDataLTSM.iaRegras.sigmoid(Wi_dot + self.bi[index_camada_oculta] + 1e-7)
                self.all_i[index_camada_oculta].append(i)

                Wo_dot = numpy.dot(self.Wo[index_camada_oculta], estado_oculto_anterior)
                o = self.modelDataLTSM.iaRegras.sigmoid(Wo_dot + self.bo[index_camada_oculta] + 1e-7)
                self.all_o[index_camada_oculta].append(o)

                Wc_dot = numpy.dot(self.Wc[index_camada_oculta], estado_oculto_anterior)
                c = self.modelDataLTSM.iaRegras.tanh(Wc_dot + self.bc[index_camada_oculta])
                self.all_c[index_camada_oculta].append(c)

                c_ = f * celulas_prev[index_camada_oculta] + i * c
                o_ = o * self.modelDataLTSM.iaRegras.tanh(c_)

                celulas_prev[index_camada_oculta] = c_
                estados_ocultos_prev[index_camada_oculta] = o_

                self.celulas_rede[index_camada_oculta].append(c_)
                self.estados_ocultos_rede[index_camada_oculta].append(o_)

            arr_saidas = []
            for index_camada_saida in range(len(self.arrCamadaSaida)):
                functionAtivacao = self.modelDataLTSM.arrFuncAtivacaoCadaSaida[index_camada_saida][0]
                saida_dot = functionAtivacao(
                    numpy.dot(self.Wv[index_camada_saida], estados_ocultos_prev[-1]) + self.bv[index_camada_saida])
                if len(saida_dot) >= 2:
                    arr_saidas.append(numpy.squeeze(saida_dot))
                else:
                    arr_saidas.append(numpy.array(saida_dot[0]))

            if isPrevisao:
                self.previsoes_rede.append(arr_saidas)
            else:
                self.saidas_rede.append(arr_saidas)

        return self.previsoes_rede if isPrevisao else self.saidas_rede

    def backward(self, entradas: list, rotulos: list):
        dWf, dWi, dWc, dWo = [], [], [], []
        dbf, dbi, dbc, dbo = [], [], [], []

        dWv, dbv = [], []

        for index_camada_oculta in range(len(self.arrNEstadosOcultos)):
            dWf.append(numpy.zeros_like(self.Wf[index_camada_oculta]))
            dWi.append(numpy.zeros_like(self.Wi[index_camada_oculta]))
            dWc.append(numpy.zeros_like(self.Wc[index_camada_oculta]))
            dWo.append(numpy.zeros_like(self.Wo[index_camada_oculta]))

            dbf.append(numpy.zeros_like(self.bf[index_camada_oculta]))
            dbi.append(numpy.zeros_like(self.bi[index_camada_oculta]))
            dbc.append(numpy.zeros_like(self.bc[index_camada_oculta]))
            dbo.append(numpy.zeros_like(self.bo[index_camada_oculta]))

        for index_camada_saida in range(len(self.arrCamadaSaida)):
            dWv.append(numpy.zeros_like(self.Wv[index_camada_saida]))
            dbv.append(numpy.zeros_like(self.bv[index_camada_saida]))

        destado_oculto_next = [numpy.zeros((i, 1)) for i in self.arrNEstadosOcultos]
        destado_celula_next = [numpy.zeros((i, 1)) for i in self.arrNEstadosOcultos]

        for index_entrada_t in reversed(range(len(entradas))):
            rotulo_t = rotulos[index_entrada_t]
            saida_t = self.saidas_rede[index_entrada_t]
            entrada_t = numpy.asarray(entradas[index_entrada_t])

            for index_camada_saida in reversed(range(len(self.arrCamadaSaida))):
                rotulo_t_resheipada = numpy.reshape(rotulo_t[index_camada_saida], (-1, 1))
                saida_t_resheipada = numpy.reshape(saida_t[index_camada_saida], (-1, 1))
                dsaida = saida_t_resheipada - rotulo_t_resheipada

                #dot_dsaida = numpy.dot(dsaida, self.estados_ocultos_rede[-1][index_entrada_t].T) * \
                             #self.modelDataLTSM.iaRegras.derivada_sigmoid(saida_t_resheipada)

                dot_dsaida = numpy.dot(dsaida, self.estados_ocultos_rede[-1][index_entrada_t].T) * self.modelDataLTSM.iaRegras.derivada_sigmoid(saida_t_resheipada)
                dWv[index_camada_saida] += dot_dsaida
                dbv[index_camada_saida] += dsaida

                dAdaWv = dWv[index_camada_saida] + self.lambdaRegAdagrad * self.Wv[index_camada_saida]
                self.Wv[index_camada_saida] -= self.txAprendizado * dAdaWv
                self.bv[index_camada_saida] -= self.txAprendizado * dbv[index_camada_saida]

                #destado_oculto_next[-1] += numpy.dot(self.Wv[index_camada_saida].T, dsaida)
                dot_destado_oculto_nect = numpy.dot(self.Wv[index_camada_saida].T, dsaida)
                destado_oculto_next[-1] += dot_destado_oculto_nect

            for index_camada_oculta in reversed(range(len(self.arrNEstadosOcultos))):
                f = self.all_f[index_camada_oculta][index_entrada_t]
                i = self.all_i[index_camada_oculta][index_entrada_t]
                o = self.all_o[index_camada_oculta][index_entrada_t]
                c = self.all_c[index_camada_oculta][index_entrada_t]
                estado_celula = self.celulas_rede[index_camada_oculta][index_entrada_t]

                if index_entrada_t > 0:
                    estado_celula_prev = self.celulas_rede[index_camada_oculta][index_entrada_t - 1]
                else:
                    estado_celula_prev = numpy.zeros_like(estado_celula)

                do = destado_oculto_next[index_camada_oculta] * self.modelDataLTSM.iaRegras.tanh(estado_celula) * \
                     self.modelDataLTSM.iaRegras.derivada_sigmoid(o)

                df = destado_celula_next[index_camada_oculta] * estado_celula_prev * \
                     self.modelDataLTSM.iaRegras.derivada_sigmoid(f)

                di = destado_celula_next[index_camada_oculta] * c * self.modelDataLTSM.iaRegras.derivada_sigmoid(i)

                dc = destado_celula_next[index_camada_oculta] * i * self.modelDataLTSM.iaRegras.derivada_tanh(c)

                destado_celula_next[index_camada_oculta] += destado_oculto_next[index_camada_oculta] * o * \
                                                            self.modelDataLTSM.iaRegras.derivada_tanh(estado_celula)

                if index_camada_oculta == 0:
                    if index_entrada_t == 0:
                        input_w = numpy.concatenate(
                            (numpy.zeros_like(self.estados_ocultos_rede[index_camada_oculta][index_entrada_t]),
                             numpy.reshape(entradas[index_entrada_t], (-1, 1))),
                            axis=0)
                    else:
                        input_w = numpy.concatenate(
                            (self.estados_ocultos_rede[index_camada_oculta][index_entrada_t - 1],
                             numpy.reshape(entradas[index_entrada_t], (-1, 1))),
                            axis=0)
                else:
                    input_w = self.estados_ocultos_rede[index_camada_oculta - 1][index_entrada_t]

                dot_dWf = numpy.dot(df, input_w.T)
                dWf[index_camada_oculta] +=  dot_dWf
                dbf[index_camada_oculta] += df

                dot_dWi = numpy.dot(di, input_w.T)
                dWi[index_camada_oculta] += dot_dWi
                dbi[index_camada_oculta] += di

                dot_dWc = numpy.dot(dc, input_w.T)
                dWc[index_camada_oculta] += dot_dWc
                dbc[index_camada_oculta] += dc

                dot_dWo = numpy.dot(do, input_w.T)
                dWo[index_camada_oculta] += dot_dWo
                dbo[index_camada_oculta] += do

                if index_camada_oculta > 0:
                    dotWf = numpy.dot(self.Wf[index_camada_oculta].T, df)
                    dotWi = numpy.dot(self.Wi[index_camada_oculta].T, di)
                    dotWc = numpy.dot(self.Wc[index_camada_oculta].T, dc)
                    dotWo = numpy.dot(self.Wf[index_camada_oculta].T, do)
                    destado_oculto_next[index_camada_oculta - 1] = dotWf + dotWi + dotWc + dotWo


                dAdaWf = dWf[index_camada_oculta] + self.lambdaRegAdagrad * self.Wf[index_camada_oculta]
                self.AdaWf[index_camada_oculta] += self.Wf[index_camada_oculta] ** 2
                self.Wf[index_camada_oculta] -= (self.txAprendizado * dAdaWf) / (numpy.sqrt(self.AdaWf[index_camada_oculta]) + 1e-7)

                dAdaWi = dWi[index_camada_oculta] + self.lambdaRegAdagrad * self.Wi[index_camada_oculta]
                self.AdaWi[index_camada_oculta] += self.Wi[index_camada_oculta] ** 2
                self.Wi[index_camada_oculta] -= (self.txAprendizado * dAdaWi) / (numpy.sqrt(self.AdaWi[index_camada_oculta]) + 1e-7)

                dAdaWc = dWc[index_camada_oculta] + self.lambdaRegAdagrad * self.Wc[index_camada_oculta]
                self.AdaWc[index_camada_oculta] += self.Wc[index_camada_oculta] ** 2
                self.Wc[index_camada_oculta] -= (self.txAprendizado * dAdaWc) / (numpy.sqrt(self.AdaWc[index_camada_oculta]) + 1e-7)

                dAdaWo = dWo[index_camada_oculta] + self.lambdaRegAdagrad * self.Wo[index_camada_oculta]
                self.AdaWo[index_camada_oculta] += self.Wo[index_camada_oculta] ** 2
                self.Wo[index_camada_oculta] -= (self.txAprendizado * dAdaWo) / (numpy.sqrt(self.AdaWo[index_camada_oculta]) + 1e-7)

                #dAdaWo = dWo[index_camada_oculta] + self.lambdaRegAdagrad * self.Wo[index_camada_oculta]
                #self.Wo[index_camada_oculta] -= self.txAprendizado * dWo[index_camada_oculta]

                dAdabf = dbf[index_camada_oculta] + self.lambdaRegAdagrad * self.bf[index_camada_oculta]
                self.bf[index_camada_oculta] -= self.txAprendizado * dAdabf

                dAdabi = dbi[index_camada_oculta] + self.lambdaRegAdagrad * self.bi[index_camada_oculta]
                self.bi[index_camada_oculta] -= self.txAprendizado * dAdabi

                dAdabc = dbc[index_camada_oculta] + self.lambdaRegAdagrad * self.bc[index_camada_oculta]
                self.bc[index_camada_oculta] -= self.txAprendizado * dAdabc

                dAdabo = dbo[index_camada_oculta] + self.lambdaRegAdagrad * self.bo[index_camada_oculta]
                self.bo[index_camada_oculta] -= self.txAprendizado * dAdabo

                #self.bo[index_camada_oculta] -= self.txAprendizado * dbo[index_camada_oculta]

    def calcular_erro(self, rotulos: list[list[list]], rotulos_originais: list[list[list]], previsoes:list[list[list]],
                      isPrintar: bool = False) -> list[list, list]:
        entropy_for_camadas = [0 for i in self.arrCamadaSaida]
        accuracy_for_camada = [0 for i in self.arrCamadaSaida]
        sum_values_accuracy = [0 for i in self.arrCamadaSaida]

        for indexClasseSaida in range(len(self.arrCamadaSaida)):
            for indexDado in range(len(rotulos)):
                saidaCamada = previsoes[indexDado][indexClasseSaida]
                rotuloOriginal = rotulos_originais[indexDado][indexClasseSaida]
                rotulo = rotulos[indexDado][indexClasseSaida]

                entropy_for_camadas[indexClasseSaida] += -numpy.mean(numpy.sum(rotulo * numpy.log(saidaCamada), axis=0))

                if len(saidaCamada) >= 2:
                    maxArg = numpy.argmax(saidaCamada)
                    if rotuloOriginal == maxArg:
                        sum_values_accuracy[indexClasseSaida] += 1
                else:
                    throbleshot = 0.5
                    y = int(saidaCamada[0] >= 0.5)
                    if y == rotuloOriginal:
                        sum_values_accuracy[indexClasseSaida] += 1

        for indexCamadaSaida in range(len(self.arrCamadaSaida)):
            accuracy_for_camada[indexCamadaSaida] = sum_values_accuracy[indexCamadaSaida] / len(rotulos_originais)
            if isPrintar:
                print(f"Entropy for camada {indexCamadaSaida}:", entropy_for_camadas[indexCamadaSaida])
                print(f"Acuracy for camada {indexCamadaSaida}:", accuracy_for_camada[indexCamadaSaida])

        return entropy_for_camadas, accuracy_for_camada
    def treinar(self) -> list[list, list]:
        nEpoca = 0
        isBrekarWhile = False

        isAttLambdRegL2 = False
        nextEpocaAttLambdRegL2 = 0

        arr_entropy = []
        entropy = [0 for i in self.arrCamadaSaida]
        media_last_entropy, media_entropy = [0 for i in self.arrCamadaSaida], [0 for i in self.arrCamadaSaida]

        arr_accuracy = []
        accuracy = [0 for i in self.arrCamadaSaida]
        media_last_accuracy, media_accuracy = [0 for i in self.arrCamadaSaida], [0 for i in self.arrCamadaSaida]

        while not isBrekarWhile:
            nEpoca += 1
            if nEpoca == self.nEpocas:
                print(entropy, accuracy)
                isBrekarWhile = True

            previsoes = self.forward(entradas=self.modelDataLTSM.arr_entradas)
            entropy, accuracy = self.calcular_erro(rotulos=self.modelDataLTSM.arr_rotulos,
                                                   rotulos_originais=self.modelDataLTSM.arr_rotulos_originais,
                                                   previsoes=previsoes)
            arr_entropy.append(entropy)
            arr_accuracy.append(accuracy)

            if nEpoca == 1:
                media_entropy, media_last_entropy = entropy, entropy
                media_accuracy, media_last_accuracy = accuracy, accuracy

            if nEpoca % 10 == 0:
                sum_entropy = numpy.sum(arr_entropy, axis=0)
                media_entropy = sum_entropy / [len(arr_entropy) for i in self.arrCamadaSaida]
                sum_accuracy = numpy.sum(arr_accuracy, axis=0)
                media_accuracy = sum_accuracy / [len(arr_accuracy) for i in self.arrCamadaSaida]

                isCaindoEntropia = ((media_entropy - media_last_entropy) / media_last_entropy) <= 0
                isCaindoAccuracy = ((media_accuracy - media_last_accuracy) / media_last_accuracy) <= 0

                arr_last_entropy, arr_entropy = arr_entropy, []
                arr_last_accuracy, arr_accuracy = arr_accuracy, []

                for indexCamadaSaida in range(len(self.arrCamadaSaida)):
                    print(f"Media entropy camada {indexCamadaSaida}:", media_entropy[indexCamadaSaida], " // ",
                          f"Media acuracy camada {indexCamadaSaida}:", media_accuracy[indexCamadaSaida])
            elif nEpoca == 1:
                print("Erro inicial: ", entropy)

            self.backward(entradas=self.modelDataLTSM.arr_entradas, rotulos=self.modelDataLTSM.arr_rotulos)

        if self.modelDataLTSM.arr_dados_prever is not None:
            previsao = self.prever(entradas=self.modelDataLTSM.arr_dados_prever)
            print("Previsão: ", previsao)

        return media_entropy, media_accuracy


    def prever(self, entradas: list[list], isPrintar: bool = False) -> list[list[list]]:
        previsao = self.forward(entradas=entradas, isPrevisao=True)
        if isPrintar:
            print(previsao)
        return previsao
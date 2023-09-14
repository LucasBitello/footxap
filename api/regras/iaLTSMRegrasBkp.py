from __future__ import annotations
from scipy.special import expit
import numpy as numpy
import random as random
from copy import deepcopy
from api.regras.iaUteisRegras import IAUteisRegras

class ModelDataLTSM:
    def __init__(self, arrEntradas: list[list], arrRotulos: list[list[list]],
                 arrDadosPrever: list[list[list]] = None, arrNameFuncAtivacaoCadaSaida: list[str] = []):

        if len(arrRotulos[0][0]) == 0:
            raise "reveja os rótulos nao é uma list[list[list]]"

        self.iaRegras = IAUteisRegras()
        self.n_epocas: int = 1000
        self.taxa_aprendizado: float = 0.08
        self.taxa_regularizacao_l2: float = 0.00000000000001

        self.arr_n_camada_oculta: list[int] = [len(arrEntradas[0])]
        self.nNeuroniosEntrada: int = len(arrEntradas[0])
        self.arrCamadasSaida: list[int] = [len(saida) for saida in arrRotulos[0]]

        self.arr_entradas: list[list] = arrEntradas
        # Lista de dados com os dados divididos em camadas lista rotulos com lista de camada
        self.arr_rotulos: list[list[list]] = arrRotulos
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
        self.lambdaRegAdagrad = modelDataLTSM.taxa_regularizacao_l2

        self.media_accuracy: float = 0.0
        self.media_entropy: float = 0.0

        self.Wf, self.Wi, self.Wc, self.Wo, self.Wv = [], [], [], [], []
        self.AdaWf, self.AdaWi, self.AdaWc, self.AdaWo, self.AdaWv = [], [], [], [], []
        self.bf, self.bi, self.bc, self.bo, self.bv = [], [], [], [], []
        self.Adabf, self.Adabi, self.Adabc, self.Adabo, self.Adabv = [], [], [], [], []

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
            nNeuroniosEntradaCamada = self.nNeuroniosEntrada if indexNNeuronios == 0 else self.arrNEstadosOcultos[indexNNeuronios - 1]
            if indexNNeuronios == 0:
                tamanho_estado_oculto = self.arrNEstadosOcultos[0]
                tamanho_estado_anterior = self.nNeuroniosEntrada #+ tamanho_estado_oculto
                arrNEstadosOcultosAnterior.append(tamanho_estado_oculto)
            else:
                tamanho_estado_oculto = self.arrNEstadosOcultos[indexNNeuronios]
                tamanho_estado_anterior = arrNEstadosOcultosAnterior[indexNNeuronios - 1]
                arrNEstadosOcultosAnterior.append(tamanho_estado_oculto)

            self.Wf.append(self.inicalizarPesosXavier(nNeuroniosEntradaCamada, (tamanho_estado_oculto, tamanho_estado_anterior), False))
            self.Wi.append(self.inicalizarPesosXavier(nNeuroniosEntradaCamada, (tamanho_estado_oculto, tamanho_estado_anterior), False))
            self.Wc.append(self.inicalizarPesosXavier(nNeuroniosEntradaCamada, (tamanho_estado_oculto, tamanho_estado_anterior), True))
            self.Wo.append(self.inicalizarPesosXavier(nNeuroniosEntradaCamada, (tamanho_estado_oculto, tamanho_estado_anterior), False))

            self.AdaWf.append(numpy.zeros_like(self.Wf[-1]))
            self.AdaWi.append(numpy.zeros_like(self.Wi[-1]))
            self.AdaWc.append(numpy.zeros_like(self.Wc[-1]))
            self.AdaWo.append(numpy.zeros_like(self.Wo[-1]))

            """self.bf.append(numpy.zeros((tamanho_estado_oculto, 1)))
            self.bi.append(numpy.zeros((tamanho_estado_oculto, 1)))
            self.bc.append(numpy.zeros((tamanho_estado_oculto, 1)))
            self.bo.append(numpy.zeros((tamanho_estado_oculto, 1)))"""

            self.bf.append(self.inicalizarPesosXavier(nNeuroniosEntradaCamada, (tamanho_estado_oculto, 1), False))
            self.bi.append(self.inicalizarPesosXavier(nNeuroniosEntradaCamada, (tamanho_estado_oculto, 1), False))
            self.bc.append(self.inicalizarPesosXavier(nNeuroniosEntradaCamada, (tamanho_estado_oculto, 1), True))
            self.bo.append(self.inicalizarPesosXavier(nNeuroniosEntradaCamada, (tamanho_estado_oculto, 1), False))

            self.Adabf.append(numpy.zeros_like(self.bf[-1]))
            self.Adabi.append(numpy.zeros_like(self.bi[-1]))
            self.Adabc.append(numpy.zeros_like(self.bc[-1]))
            self.Adabo.append(numpy.zeros_like(self.bo[-1]))

        for nNeuroniosSaida in self.arrCamadaSaida:
            sumNeuronios = self.nNeuroniosEntrada + nNeuroniosSaida
            self.Wv.append(self.inicalizarPesosXavier(sumNeuronios, (nNeuroniosSaida, self.arrNEstadosOcultos[-1]), False))
            self.AdaWv.append(numpy.zeros_like(self.Wv[-1]))

            self.bv.append(self.inicalizarPesosXavier(sumNeuronios, (nNeuroniosSaida, 1), False))
            self.Adabv.append(numpy.zeros_like(self.bv[-1]))

    def inicalizarPesosXavier(self, nItens: int, tupleDim: tuple, isScalaMenorZero: bool = True,
                              isReturnMatriz: bool = True) -> list:
        initScale = -1 if isScalaMenorZero else 0
        endScale = 1 if isScalaMenorZero else 1
        if isReturnMatriz:
            arrXavier = numpy.random.uniform(initScale, endScale, tupleDim)
        else:
            arrXavier = numpy.random.uniform(initScale, endScale, tupleDim[0])

        return arrXavier

    def init_new_epoca(self):
        self.all_f = [[] for _ in range(len(self.arrNEstadosOcultos))]
        self.all_i = [[] for _ in range(len(self.arrNEstadosOcultos))]
        self.all_o = [[] for _ in range(len(self.arrNEstadosOcultos))]
        self.all_c = [[] for _ in range(len(self.arrNEstadosOcultos))]

    def forward(self, entradas: list, isPrevisao: bool = False, initNewEpoca: bool = True, celulas_prev: list = None,
                estados_ocultos_prev: list = None) -> list[list[list]]:
        if initNewEpoca:
            self.init_new_epoca()

        if celulas_prev is None:
            celulas_prev = [numpy.zeros((i, 1)) for i in self.arrNEstadosOcultos]

        if estados_ocultos_prev is None:
            estados_ocultos_prev = [numpy.zeros((i, 1)) for i in self.arrNEstadosOcultos]

        saidas_rede = []
        celulas_rede: list = [[] for _ in self.arrNEstadosOcultos]
        estados_ocultos_rede: list = [[] for _ in self.arrNEstadosOcultos]



        for index_entrada_t in range(len(entradas)):
            entrada_t = numpy.transpose([entradas[index_entrada_t]])
            # entrada_t = numpy.concatenate((estados_ocultos_prev[0], entrada_t), axis=0)

            for index_camada_oculta in range(len(self.arrNEstadosOcultos)):
                if index_camada_oculta == 0:
                    estado_oculto_anterior = entrada_t
                else:
                    estado_oculto_anterior = estados_ocultos_prev[index_camada_oculta - 1]

                Wf_dot = numpy.dot(self.Wf[index_camada_oculta], estado_oculto_anterior) #+ self.bf[index_camada_oculta]
                f = self.modelDataLTSM.iaRegras.sigmoid(Wf_dot)
                if initNewEpoca:
                    self.all_f[index_camada_oculta].append(f)

                Wi_dot = numpy.dot(self.Wi[index_camada_oculta], estado_oculto_anterior) #+ self.bi[index_camada_oculta]
                i = self.modelDataLTSM.iaRegras.sigmoid(Wi_dot)
                if initNewEpoca:
                    self.all_i[index_camada_oculta].append(i)

                Wo_dot = numpy.dot(self.Wo[index_camada_oculta], estado_oculto_anterior) #+ self.bo[index_camada_oculta]
                o = self.modelDataLTSM.iaRegras.sigmoid(Wo_dot)
                if initNewEpoca:
                    self.all_o[index_camada_oculta].append(o)

                Wc_dot = numpy.dot(self.Wc[index_camada_oculta], estado_oculto_anterior) #+ self.bc[index_camada_oculta]
                c = self.modelDataLTSM.iaRegras.tanh(Wc_dot)
                if initNewEpoca:
                    self.all_c[index_camada_oculta].append(c)

                c_ = f * celulas_prev[index_camada_oculta] + i * c
                o_ = o * self.modelDataLTSM.iaRegras.tanh(c_)

                celulas_prev[index_camada_oculta] = c_
                estados_ocultos_prev[index_camada_oculta] = o_

                celulas_rede[index_camada_oculta].append(c_)
                estados_ocultos_rede[index_camada_oculta].append(o_)

            arr_saidas = []
            for index_camada_saida in range(len(self.arrCamadaSaida)):
                functionAtivacao = self.modelDataLTSM.arrFuncAtivacaoCadaSaida[index_camada_saida][0]
                saida_dot = numpy.dot(self.Wv[index_camada_saida], estados_ocultos_prev[-1]) #+ self.bv[index_camada_saida]
                saida_ativ = functionAtivacao(saida_dot)

                arr_saidas.append(saida_ativ)

            saidas_rede.append(arr_saidas)

        return saidas_rede, celulas_rede, estados_ocultos_rede, celulas_prev, estados_ocultos_prev

    def backward(self, entradas: list, saidas: list, rotulos: list, celulas_rede: list, estados_ocultos_rede: list):
        dWf, dWi, dWc, dWo = [], [], [], []
        dAdaWf, dAdaWi, dAdaWc, dAdaWo = [], [], [], []
        dbf, dbi, dbc, dbo = [], [], [], []
        dAdabf, dAdabi, dAdabc, dAdabo = [], [], [], []

        dWv, dbv = [], []
        dAdaWv, dAdabv = [], []

        for index_camada_oculta in range(len(self.arrNEstadosOcultos)):
            dWf.append(numpy.zeros_like(self.Wf[index_camada_oculta]))
            dAdaWf.append(numpy.zeros_like(dWf[-1]))
            dWi.append(numpy.zeros_like(self.Wi[index_camada_oculta]))
            dAdaWi.append(numpy.zeros_like(dWi[-1]))
            dWc.append(numpy.zeros_like(self.Wc[index_camada_oculta]))
            dAdaWc.append(numpy.zeros_like(dWc[-1]))
            dWo.append(numpy.zeros_like(self.Wo[index_camada_oculta]))
            dAdaWo.append(numpy.zeros_like(dWo[-1]))

            dbf.append(numpy.zeros_like(self.bf[index_camada_oculta]))
            dAdabf.append(numpy.zeros_like(dbf[-1]))
            dbi.append(numpy.zeros_like(self.bi[index_camada_oculta]))
            dAdabi.append(numpy.zeros_like(dbi[-1]))
            dbc.append(numpy.zeros_like(self.bc[index_camada_oculta]))
            dAdabc.append(numpy.zeros_like(dbc[-1]))
            dbo.append(numpy.zeros_like(self.bo[index_camada_oculta]))
            dAdabo.append(numpy.zeros_like(dbo[-1]))

        for index_camada_saida in range(len(self.arrCamadaSaida)):
            dWv.append(numpy.zeros_like(self.Wv[index_camada_saida]))
            dAdaWv.append(numpy.zeros_like(dWv[-1]))
            dbv.append(numpy.zeros_like(self.bv[index_camada_saida]))
            dAdabv.append(numpy.zeros_like(dbv[-1]))

        destado_oculto_next = [numpy.zeros((i, 1)) for i in self.arrNEstadosOcultos]
        destado_celula_next = [numpy.zeros((i, 1)) for i in self.arrNEstadosOcultos]
        do_ = [numpy.zeros((i, 1)) for i in self.arrNEstadosOcultos]

        for index_entrada_t in reversed(range(len(entradas))):
            rotulo = rotulos[index_entrada_t]
            saida = saidas[index_entrada_t]

            dsaida_s = numpy.zeros((self.arrNEstadosOcultos[-1], 1))
            for index_camada_saida in reversed(range(len(self.arrCamadaSaida))):
                funcDerivada = self.modelDataLTSM.arrFuncAtivacaoCadaSaida[index_camada_saida][1]
                funcDerivadaOculta = self.modelDataLTSM.iaRegras.derivada_tanh

                estado_oculto_ultima_camada_t = estados_ocultos_rede[-1][index_entrada_t]
                estado_celula_ultima_camada_t = celulas_rede[-1][index_entrada_t]

                rotulo_st = numpy.asarray(numpy.transpose([rotulo[index_camada_saida]]))
                saida_st = numpy.asarray(saida[index_camada_saida])

                erro = saida_st - rotulo_st
                dsaida = erro
                dot_dsaida = numpy.dot(dsaida, estado_oculto_ultima_camada_t.T) * funcDerivada(saida_st)

                dWv[index_camada_saida] += dot_dsaida
                dbv[index_camada_saida] += erro

                dot_destado_oculto_next = numpy.dot(self.Wv[index_camada_saida].T, dsaida)
                dsaida_s += dot_destado_oculto_next

                dAdaWv[index_camada_saida] += dWv[index_camada_saida] + self.lambdaRegAdagrad * self.Wv[index_camada_saida]
                self.AdaWv[index_camada_saida] += dAdaWv[index_camada_saida] ** 2

                dAdabv[index_camada_saida] += dbv[index_camada_saida] + self.lambdaRegAdagrad * self.bv[index_camada_saida]
                self.Adabv[index_camada_saida] += dAdabv[index_camada_saida] ** 2

                if index_entrada_t >= 0:
                    self.Wv[index_camada_saida] -= (self.txAprendizado * dAdaWv[index_camada_saida]) / \
                                                   (numpy.sqrt(self.AdaWv[index_camada_saida]) + 1e-7)
                    self.bv[index_camada_saida] -= (self.txAprendizado * dAdabv[index_camada_saida]) / \
                                                   (numpy.sqrt(self.Adabv[index_camada_saida]) + 1e-7)

            destado_oculto_next[-1] = dsaida_s
            for index_camada_oculta in reversed(range(len(self.arrNEstadosOcultos))):
                f = self.all_f[index_camada_oculta][index_entrada_t]
                i = self.all_i[index_camada_oculta][index_entrada_t]
                o = self.all_o[index_camada_oculta][index_entrada_t]
                c = self.all_c[index_camada_oculta][index_entrada_t]
                estado_celula = celulas_rede[index_camada_oculta][index_entrada_t]
                estado_oculto = estados_ocultos_rede[index_camada_oculta][index_entrada_t]

                dc_ = numpy.clip((destado_oculto_next[index_camada_oculta] * o *
                                  self.modelDataLTSM.iaRegras.derivada_tanh(estado_celula)) +
                                 destado_celula_next[index_camada_oculta], -5, 5)

                if index_entrada_t > 0:
                    estado_celula_prev = celulas_rede[index_camada_oculta][index_entrada_t - 1]
                    estado_oculto_prev = estados_ocultos_rede[index_camada_oculta][index_entrada_t - 1]
                else:
                    estado_celula_prev = numpy.zeros_like(estado_celula)
                    estado_oculto_prev = numpy.zeros_like(estado_oculto)

                do = destado_oculto_next[index_camada_oculta] * estado_oculto

                df = dc_ * estado_celula_prev

                di = dc_ * c

                dc = dc_ * i

                if index_camada_oculta == 0:
                    if index_entrada_t == 0:
                        zeros = numpy.zeros_like(estado_oculto_prev)
                        input_w = numpy.transpose([entradas[index_entrada_t]])
                        # input_w = numpy.concatenate((zeros, input_w), axis=0)
                    else:
                        input_w = numpy.transpose([entradas[index_entrada_t]])
                        # input_w = numpy.concatenate((estado_oculto_prev, input_w), axis=0)
                else:
                    input_w = estados_ocultos_rede[index_camada_oculta - 1][index_entrada_t]

                dot_dWf = numpy.dot(df, numpy.transpose(input_w))
                dWf[index_camada_oculta] += dot_dWf
                dbf[index_camada_oculta] += df

                dot_dWi = numpy.dot(di, numpy.transpose(input_w))
                dWi[index_camada_oculta] += dot_dWi
                dbi[index_camada_oculta] += di

                dot_dWc = numpy.dot(dc, numpy.transpose(input_w))
                dWc[index_camada_oculta] += dot_dWc
                dbc[index_camada_oculta] += dc

                dot_dWo = numpy.dot(do, numpy.transpose(input_w))
                dWo[index_camada_oculta] += dot_dWo
                dbo[index_camada_oculta] += do

                dotWf = numpy.dot(self.Wf[index_camada_oculta].T, df)
                dotWi = numpy.dot(self.Wi[index_camada_oculta].T, di)
                dotWc = numpy.dot(self.Wc[index_camada_oculta].T, dc)
                dotWo = numpy.dot(self.Wf[index_camada_oculta].T, do)

                dDtos = dotWf + dotWi + dotWc + dotWo
                if index_camada_oculta > 0:
                    destado_oculto_next[index_camada_oculta - 1] = dDtos
                destado_celula_next[index_camada_oculta] = dc_

                dAdaWf[index_camada_oculta] += dWf[index_camada_oculta] + self.lambdaRegAdagrad * self.Wf[index_camada_oculta]
                self.AdaWf[index_camada_oculta] += dAdaWf[index_camada_oculta] ** 2

                dAdabf[index_camada_oculta] += dbf[index_camada_oculta] + self.lambdaRegAdagrad * self.bf[index_camada_oculta]
                self.Adabf[index_camada_oculta] += dAdabf[index_camada_oculta] ** 2

                dAdaWi[index_camada_oculta] += dWi[index_camada_oculta] + self.lambdaRegAdagrad * self.Wi[index_camada_oculta]
                self.AdaWi[index_camada_oculta] += dAdaWi[index_camada_oculta] ** 2

                dAdabi[index_camada_oculta] += dbi[index_camada_oculta] + self.lambdaRegAdagrad * self.bi[index_camada_oculta]
                self.Adabi[index_camada_oculta] += dAdabi[index_camada_oculta] ** 2

                dAdaWc[index_camada_oculta] += dWc[index_camada_oculta] + self.lambdaRegAdagrad * self.Wc[index_camada_oculta]
                self.AdaWc[index_camada_oculta] += dAdaWc[index_camada_oculta] ** 2

                dAdabc[index_camada_oculta] += dbc[index_camada_oculta] + self.lambdaRegAdagrad * self.bc[index_camada_oculta]
                self.Adabc[index_camada_oculta] += dAdabc[index_camada_oculta] ** 2

                dAdaWo[index_camada_oculta] += dWo[index_camada_oculta] + self.lambdaRegAdagrad * self.Wo[index_camada_oculta]
                self.AdaWo[index_camada_oculta] += dAdaWo[index_camada_oculta] ** 2

                dAdabo[index_camada_oculta] += dbo[index_camada_oculta] + self.lambdaRegAdagrad * self.bo[index_camada_oculta]
                self.Adabo[index_camada_oculta] += dAdabo[index_camada_oculta] ** 2

                if index_entrada_t >= 0:
                    self.Wf[index_camada_oculta] -= (self.txAprendizado * dAdaWf[index_camada_oculta]) / (numpy.sqrt(self.AdaWf[index_camada_oculta]) + 1e-7)
                    self.Wi[index_camada_oculta] -= (self.txAprendizado * dAdaWi[index_camada_oculta]) / (numpy.sqrt(self.AdaWi[index_camada_oculta]) + 1e-7)
                    self.Wc[index_camada_oculta] -= (self.txAprendizado * dAdaWc[index_camada_oculta]) / (numpy.sqrt(self.AdaWc[index_camada_oculta]) + 1e-7)
                    self.Wo[index_camada_oculta] -= (self.txAprendizado * dAdaWo[index_camada_oculta]) / (numpy.sqrt(self.AdaWo[index_camada_oculta]) + 1e-7)
                    self.bf[index_camada_oculta] -= (self.txAprendizado * dAdabf[index_camada_oculta]) / (numpy.sqrt(self.Adabf[index_camada_oculta]) + 1e-7)
                    self.bi[index_camada_oculta] -= (self.txAprendizado * dAdabi[index_camada_oculta]) / (numpy.sqrt(self.Adabi[index_camada_oculta]) + 1e-7)
                    self.bc[index_camada_oculta] -= (self.txAprendizado * dAdabc[index_camada_oculta]) / (numpy.sqrt(self.Adabc[index_camada_oculta]) + 1e-7)
                    self.bo[index_camada_oculta] -= (self.txAprendizado * dAdabo[index_camada_oculta]) / (numpy.sqrt(self.Adabo[index_camada_oculta]) + 1e-7)

                # dAdaWo = dWo[index_camada_oculta] + self.lambdaRegAdagrad * self.Wo[index_camada_oculta]
                # self.Wo[index_camada_oculta] -= self.txAprendizado * dWo[index_camada_oculta]
                #self.bo[index_camada_oculta] -= self.txAprendizado * dbo[index_camada_oculta]

    def calcular_erro(self, rotulos: list[list[list]], arrPrevisoes: list[list[list[list]]], isPrintar: bool = True) -> \
    list[list, list]:
        arrEntropy, arrAcurracy = [], []

        for index_previsoes in range(len(arrPrevisoes)):
            previsoes = arrPrevisoes[index_previsoes]
            entropy_for_camadas = [0 for i in self.arrCamadaSaida]
            accuracy_for_camada = [0 for i in self.arrCamadaSaida]
            sum_values_accuracy = [0 for i in self.arrCamadaSaida]

            for indexClasseSaida in range(len(self.arrCamadaSaida)):
                for indexDado in range(len(previsoes)):
                    saidaCamada = numpy.reshape(previsoes[indexDado][indexClasseSaida], (1, -1))[0]
                    rotulo = rotulos[indexDado][indexClasseSaida]

                    dotEntropy = rotulo * numpy.log(saidaCamada)
                    entropy_for_camadas[indexClasseSaida] += -numpy.sum(dotEntropy, axis=0)

                    if len(saidaCamada) >= 2:
                        maxArgSaida = numpy.argmax(saidaCamada)
                        maxArgRotulo = numpy.argmax(rotulo)
                        if maxArgRotulo == maxArgSaida:
                            sum_values_accuracy[indexClasseSaida] += 1
                    else:
                        throbleshot = 0.5
                        y = int(saidaCamada[0] >= throbleshot)
                        if y == rotulo[0]:
                            sum_values_accuracy[indexClasseSaida] += 1

            msgEntropy = "Entropy for camada "
            msgAcurracy = "Acurracy for camada "
            for indexCamadaSaida in range(len(self.arrCamadaSaida)):
                accuracy_for_camada[indexCamadaSaida] = sum_values_accuracy[indexCamadaSaida] / len(rotulos)

                if isPrintar:
                    #msgEntropy += f" [{indexCamadaSaida}: {entropy_for_camadas[indexCamadaSaida]:.5f}], "
                    #msgAcurracy += f" [{indexCamadaSaida}: {accuracy_for_camada[indexCamadaSaida]:.5f}], "
                    pass

            if isPrintar:
                #print(msgEntropy, msgAcurracy)
                #print("######################")
                pass

            arrEntropy.append(entropy_for_camadas)
            arrAcurracy.append(accuracy_for_camada)

        mediaEntropy = numpy.sum(arrEntropy, axis=0) / len(arrEntropy)
        mediaAcurracy = numpy.sum(arrAcurracy, axis=0) / len(arrAcurracy)
        msgMediaEntropy = " Media entropy for camada"
        msgMediaAcurracy = "Media acuracy for camada"

        for indexCamadaSaida in range(len(self.arrCamadaSaida)):
            accuracy_for_camada[indexCamadaSaida] = sum_values_accuracy[indexCamadaSaida] / len(rotulos)
            msgMediaEntropy += f" [{indexCamadaSaida}: {mediaEntropy[indexCamadaSaida]:.5f}], "
            msgMediaAcurracy += f" [{indexCamadaSaida}: {mediaAcurracy[indexCamadaSaida]:.5f}], "

        if isPrintar:
            print(msgMediaEntropy, "\n", msgMediaAcurracy)
            print("##########")

        return mediaEntropy, mediaAcurracy

    def treinar(self, isBrekarPorEpocas: bool = True, isAtualizarPesos: bool = True, qtdeDadoValidar: int = 2,
                n_folds: int = 5, isRecursiva: bool = False) -> list[list, list]:

        nEpoca = 0
        isBrekarWhile = False

        isAttLambdRegL2 = False
        nextEpocaAttLambdRegL2 = 0

        arrEntradas = deepcopy(self.modelDataLTSM.arr_entradas)
        arrRotulos = deepcopy(self.modelDataLTSM.arr_rotulos)

        arrArrSaidas = []

        arrEntradas_t = arrEntradas
        arrRotulos_t = arrRotulos


        if qtdeDadoValidar >= 1:
            arrEntradas_v = arrEntradas[-qtdeDadoValidar:]
            arrRotulos_v = arrRotulos[-qtdeDadoValidar:]

            for i in range(qtdeDadoValidar):
                arrEntradas.pop()
                arrRotulos.pop()
        else:
            arrEntradas_v = arrEntradas[-int(len(arrEntradas_t)):]
            arrRotulos_v = arrRotulos[-int(len(arrRotulos_t)):]

        isAcuraciaBoa = False
        while not isBrekarWhile:
            nEpoca += 1
            previsoes, celulas, estados_ocultos, celulas_prev, estados_ocultos_prev = self.forward(entradas=arrEntradas_t)
            arrArrSaidas.append(previsoes)

            if True:
                if nEpoca % 100 == 0:
                    print(nEpoca, "de", self.nEpocas)

                media_entropy, media_accuracy = self.calcular_erro(rotulos=arrRotulos_t, arrPrevisoes=arrArrSaidas, isPrintar=nEpoca % 100 == 0)
                arrArrSaidas = []

                if sum(media_accuracy) / len(media_accuracy) >= 0.93:
                    isAcuraciaBoa = True
                else:
                    isAcuraciaBoa = False

                if sum(media_accuracy) / len(media_accuracy) >= 1 and not isRecursiva and nEpoca >= int(self.nEpocas):
                    print("Não foi possível encontrar um bom resultado. Chegaram a 1")
                    return False

            if isAcuraciaBoa:
                previsoes_v = self.forward(entradas=arrEntradas_v, initNewEpoca=False, celulas_prev=celulas_prev if qtdeDadoValidar >= 1 else None,
                                           estados_ocultos_prev=estados_ocultos_prev if qtdeDadoValidar >= 1 else None)[0]

                if nEpoca % 10 == 0:
                    print("############ Validação #############")

                media_entropy_v, media_accuracy_v = self.calcular_erro(rotulos=arrRotulos_v, arrPrevisoes=[previsoes_v],
                                                                       isPrintar=nEpoca % 10 == 0)

                if sum(media_accuracy_v) / len(media_accuracy_v) >= 1:
                    if self.modelDataLTSM.arr_dados_prever is not None:
                        previsoes_v, celulas_v, estados_ocultos_v, celulas_prev_v, estados_ocultos_prev_v = \
                            self.forward(entradas=arrEntradas_v, initNewEpoca=False, celulas_prev=celulas_prev if qtdeDadoValidar >= 1 else None,
                            estados_ocultos_prev=estados_ocultos_prev if qtdeDadoValidar >= 1 else None)

                        previsao = self.forward(entradas=self.modelDataLTSM.arr_dados_prever, initNewEpoca=False,
                                                estados_ocultos_prev=estados_ocultos_prev_v,
                                                celulas_prev=celulas_prev_v)[0]
                        for prev in previsao:
                            print("Previsão: ", [a for a in prev])

                    break

            if isBrekarPorEpocas and nEpoca == self.nEpocas:
                print("Não foi possível encontrar um bom resultado. 3")
                if not isAtualizarPesos:
                    break
                else:
                    return False

            if isAtualizarPesos:
                self.backward(entradas=arrEntradas_t, rotulos=arrRotulos_t, celulas_rede=celulas,
                              estados_ocultos_rede=estados_ocultos, saidas=previsoes)

        self.media_entropy = media_entropy
        self.media_accuracy = media_accuracy

        return media_entropy, media_accuracy


    def prever(self, entradas: list[list], isPrintar: bool = False) -> list[list[list]]:
        previsao = self.forward(entradas=entradas, isPrevisao=True)[0]
        if isPrintar:
            print("########## Previsao ##########")
            for prev in previsao:
                print(prev)
            print("########################")
        return previsao
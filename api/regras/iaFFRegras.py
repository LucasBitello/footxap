import numpy as numpy

from api.regras.iaUteisRegras import IAUteisRegras
from sklearn.model_selection import KFold
class ModelDataFF:
    def __init__(self, arrEntradas: list[list], arrRotulos: list[list[list]],
                 arrDadosPrever: list[list] = None, arrNameFuncAtivacaoCadaOculta: list[str] = [],
                 arrNameFuncAtivacaoCadaSaida: list[str] = []):

        if len(arrRotulos[0][0]) == 0:
            raise "reveja os rótulos nao é uma list[list[list]]"

        self.iaRegras = IAUteisRegras()
        self.n_epocas: int = 1500
        self.taxa_aprendizado: float = 0.05
        self.taxa_regularização_l2: float = 0.001

        self.arr_n_camada_oculta: list[int] = [len(arrEntradas[0])]
        self.nNeuroniosEntrada: int = len(arrEntradas[0])
        self.arrCamadasSaida: list[int] = [len(saida) for saida in arrRotulos[0]]

        self.arr_entradas: list[list] = arrEntradas
        # Lista de dados com os dados divididos em camadas lista rotulos com lista de camada
        self.arr_rotulos: list[list[list]] = arrRotulos
        self.arr_dados_prever: list[list] = arrDadosPrever
        self.arrFuncAtivacaoCadaOculta: list[list[any, any]] = []
        self.arrFuncAtivacaoCadaSaida: list[list[any, any]] = []

        if len(arrNameFuncAtivacaoCadaSaida) == 0:
            for camadaSaida in self.arr_rotulos[0]:
                arrNameFuncAtivacaoCadaSaida.append("sigmoid")

        if len(arrNameFuncAtivacaoCadaSaida) == 1 and len(self.arrCamadasSaida) >= 2:
            for indexCamadaSaida in range(len(self.arrCamadasSaida)):
                if indexCamadaSaida >= 1:
                    arrNameFuncAtivacaoCadaSaida.append(arrNameFuncAtivacaoCadaSaida[0])

        if len(arrNameFuncAtivacaoCadaOculta) == 0:
            for camadaSaida in self.arr_n_camada_oculta:
                arrNameFuncAtivacaoCadaOculta.append("sigmoid")

        if len(arrNameFuncAtivacaoCadaOculta) == 1 and len(self.arr_n_camada_oculta) >= 2:
            for indexCamada in range(len(self.arr_n_camada_oculta)):
                if indexCamada >= 1:
                    arrNameFuncAtivacaoCadaOculta.append(arrNameFuncAtivacaoCadaOculta[0])

        for name in arrNameFuncAtivacaoCadaOculta:
            arrFunctActivDeriv = []
            if name == "softmax":
                arrFunctActivDeriv = [self.iaRegras.softmax, self.iaRegras.derivada_softmax]
                self.arrFuncAtivacaoCadaOculta.append(arrFunctActivDeriv)
            elif name == "sigmoid":
                arrFunctActivDeriv = [self.iaRegras.sigmoid, self.iaRegras.derivada_sigmoid]
                self.arrFuncAtivacaoCadaOculta.append(arrFunctActivDeriv)
            else:
                raise "Functions de ativação nao edfinidos"

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


class FF:
    def __init__(self, modelDataFF: ModelDataFF):
        self.modelDataFF = modelDataFF
        self.nEpocas = modelDataFF.n_epocas
        self.txAprendizado = modelDataFF.taxa_aprendizado
        self.arrCamadaSaida = modelDataFF.arrCamadasSaida
        self.nNeuroniosEntrada = modelDataFF.nNeuroniosEntrada
        self.arrNEstadosOcultos = modelDataFF.arr_n_camada_oculta
        self.lambdaRegAdagrad = modelDataFF.taxa_regularização_l2

        self.media_accuracy: float = 0.0
        self.media_entropy: float = 0.0

        self.Wh, self.Wo = [], []
        self.AdaWh, self.AdaWo = [], []
        self.bh, self.bo = [], []
        self.Adabh, self.Adabo = [], []

        self.saidas_rede: list = []
        self.previsoes_rede: list = []

        arrNEstadosOcultosAnterior = []
        for indexNNeuronios in range(len(self.arrNEstadosOcultos)):
            nNeuroniosEntradaCamada = self.nNeuroniosEntrada if indexNNeuronios == 0 else \
                self.arrNEstadosOcultos[indexNNeuronios - 1]
            if indexNNeuronios == 0:
                tamanho_estado_oculto = self.arrNEstadosOcultos[0]
                tamanho_estado_anterior = self.nNeuroniosEntrada
                arrNEstadosOcultosAnterior.append(tamanho_estado_oculto)
            else:
                tamanho_estado_oculto = self.arrNEstadosOcultos[indexNNeuronios]
                tamanho_estado_anterior = arrNEstadosOcultosAnterior[indexNNeuronios - 1]
                arrNEstadosOcultosAnterior.append(tamanho_estado_oculto)

            self.Wh.append(self.inicalizarPesosXavier(nNeuroniosEntradaCamada,
                                                      (tamanho_estado_anterior, tamanho_estado_oculto), False))

            self.AdaWh.append(numpy.zeros_like(self.Wh[-1]))

            self.bh.append(
                self.inicalizarPesosXavier(nNeuroniosEntradaCamada, (tamanho_estado_oculto, 1), False, isReturnMatriz=False))

            self.Adabh.append(numpy.zeros_like(self.bh[-1]))

        for nNeuroniosSaida in self.arrCamadaSaida:
            sumNeuronios = self.nNeuroniosEntrada + nNeuroniosSaida
            self.Wo.append(
                self.inicalizarPesosXavier(sumNeuronios, (self.arrNEstadosOcultos[-1],nNeuroniosSaida), False))
            self.AdaWo.append(numpy.zeros_like(self.Wo[-1]))

            self.bo.append(self.inicalizarPesosXavier(sumNeuronios, (nNeuroniosSaida, 1), False, isReturnMatriz=False))
            self.Adabo.append(numpy.zeros_like(self.bo[-1]))

    def inicalizarPesosXavier(self, nItens: int, tupleDim: tuple, isScalaMenorZero: bool = True,
                              isReturnMatriz: bool = True) -> list:
        initScale = -numpy.sqrt(2 / nItens) if isScalaMenorZero else 0
        endScale = 1

        if isReturnMatriz:
            arrXavier = numpy.random.uniform(initScale, endScale, tupleDim)
        else:
            arrXavier = numpy.random.uniform(initScale, endScale, tupleDim[0])

        return arrXavier

    def forward(self, entradas: list[list]) -> (list[list], list[list[list]]):
        ativacoesOcultas = []
        pMatricialOcultos = []
        arrSaidas = []
        pMatricialSaidas = []

        for indexCamada in range(len(self.Wh)):
            funcAtivacao = self.modelDataFF.arrFuncAtivacaoCadaOculta[indexCamada][0]

            estado_oculto_anterior = numpy.transpose(entradas) if indexCamada == 0 else ativacoesOcultas[-1]
            p_matricial = numpy.dot(self.Wh[indexCamada].T, estado_oculto_anterior)
            pMatricialOcultos.append(p_matricial)

            ativacao = funcAtivacao(p_matricial)
            ativacoesOcultas.append(ativacao)


        for indexCamadaSaida in range(len(self.arrCamadaSaida)):
            funcAtivacao = self.modelDataFF.arrFuncAtivacaoCadaSaida[indexCamadaSaida][0]

            p_matricial = numpy.dot(self.Wo[indexCamadaSaida].T, ativacoesOcultas[-1])
            pMatricialSaidas.append(p_matricial)

            ativacao = funcAtivacao(p_matricial)
            arrSaidas.append(ativacao)

        return pMatricialOcultos, ativacoesOcultas, pMatricialSaidas, arrSaidas


    def backward(self, entradas: list[list], estados_ocultos: list[list], saidas_rede: list[list[list]],
                 rotulos:list[list[list]], pMatricalOcultas: list[list], pMatricialSaidas: list[list], acertos: list):
        dWh, dWo = [], []
        dAdaWh, dAdaWo = [], []
        dbh, dbo = [], []
        dAdabh, dAdabo = [], []

        for index_camada_oculta in range(len(self.arrNEstadosOcultos)):
            dWh.append(numpy.zeros_like(self.Wh[index_camada_oculta]))
            dAdaWh.append(numpy.zeros_like(dWh[-1]))
            dbh.append(numpy.zeros_like(self.bh[index_camada_oculta]))
            dAdabh.append(numpy.zeros_like(dbh[-1]))


        for index_camada_saida in range(len(self.arrCamadaSaida)):
            dWo.append(numpy.zeros_like(self.Wo[index_camada_oculta]))
            dAdaWo.append(numpy.zeros_like(dWo[-1]))
            dbo.append(numpy.zeros_like(self.bo[index_camada_oculta]))
            dAdabo.append(numpy.zeros_like(dbo[-1]))

        delta_h = numpy.zeros(self.arrNEstadosOcultos[-1])

        for indexSaida in range(len(saidas_rede)):
            funcDerivada = self.modelDataFF.arrFuncAtivacaoCadaSaida[indexSaida][1]
            saidas_norm = saidas_rede[indexSaida]
            rotulos_norm = numpy.transpose([rotulo[indexSaida] for rotulo in rotulos])
            erro_o = saidas_norm - rotulos_norm
            delta_o = erro_o * funcDerivada(saidas_rede[indexSaida])

            dWo[indexSaida] = numpy.dot(estados_ocultos[-1], delta_o.T)
            dbo[indexSaida] = numpy.sum(erro_o, axis=1)

            derivAtivacaoOculta = self.modelDataFF.arrFuncAtivacaoCadaOculta[-1][1]
            if indexSaida == 0:
                delta_h = numpy.dot(self.Wo[indexSaida], erro_o) * derivAtivacaoOculta(estados_ocultos[-1])
            else:
                delta_h += numpy.dot(self.Wo[indexSaida], erro_o) * derivAtivacaoOculta(estados_ocultos[-1])

            dAdaWo[indexSaida] = dWo[indexSaida] + self.lambdaRegAdagrad * self.Wo[indexSaida]
            self.AdaWo[indexSaida] = dAdaWo[indexSaida] ** 2
            self.Wo[indexSaida] -= (self.txAprendizado * dAdaWo[indexSaida]) / (numpy.sqrt(self.AdaWo[indexSaida]) + 1e-7)

            dAdabo[indexSaida] = dbo[indexSaida] + self.lambdaRegAdagrad * self.bo[indexSaida]
            self.Adabo[indexSaida] += dAdabo[indexSaida] ** 2
            self.bo[indexSaida] -= (self.txAprendizado * dAdabo[indexSaida]) / (numpy.sqrt(self.Adabo[indexSaida]) + 1e-7)

        for indexOculto in range(len(self.arrNEstadosOcultos) - 1, -1, -1):
            derivAtivacaoOculta = self.modelDataFF.arrFuncAtivacaoCadaOculta[-1][1]

            if indexOculto == 0:
                estado_oculto_anterior = numpy.transpose(entradas)
            else:
                estado_oculto_anterior = estados_ocultos[indexOculto - 1]

            dWh[indexOculto] = numpy.dot(estado_oculto_anterior, numpy.transpose(delta_h))
            dbh[indexOculto] = numpy.sum(delta_h, axis=1)

            if indexOculto > 0:
                delta_h = numpy.dot(self.Wh[indexOculto], delta_h) * derivAtivacaoOculta(estados_ocultos[indexOculto - 1])

            dAdaWh[indexOculto] = dWh[indexOculto] + self.lambdaRegAdagrad * self.Wh[indexOculto]
            self.AdaWh[indexOculto] = dAdaWh[indexOculto] ** 2
            self.Wh[indexOculto] -= (self.txAprendizado * dAdaWh[indexOculto]) / (numpy.sqrt(self.AdaWh[indexOculto]) + 1e-7)

            dAdabh[indexOculto] = dbh[indexOculto] + self.lambdaRegAdagrad * self.bh[indexOculto]
            self.Adabh[indexOculto] += dAdabh[indexOculto] ** 2
            self.bh[indexOculto] -= (self.txAprendizado * dAdabh[indexOculto]) / (numpy.sqrt(self.Adabh[indexOculto]) + 1e-7)


    def calcular_erro(self, rotulos: list[list[list]], arrPrevisoes:list[list[list[list]]], isPrintar: bool = True) -> list[list, list, list]:
        arrEntropy, arrAcurracy = [], []
        acertos = []

        for i_previsoes in range(len(arrPrevisoes)):
            entropy_for_camadas = [0 for i in self.arrCamadaSaida]
            accuracy_for_camada = [0 for i in self.arrCamadaSaida]
            sum_values_accuracy = [0 for i in self.arrCamadaSaida]
            arrAcertos = [[] for i in self.arrCamadaSaida]

            for indexClasseSaida in range(len(self.arrCamadaSaida)):
                for indexDado in range(len(rotulos[i_previsoes])):
                    saidaCamada = arrPrevisoes[i_previsoes][indexClasseSaida][:, indexDado]
                    rotulo = rotulos[i_previsoes][indexDado][indexClasseSaida]

                    dotEntropy = rotulo * numpy.log(saidaCamada)
                    entropy_for_camadas[indexClasseSaida] += -numpy.sum(dotEntropy, axis=0)

                    if len(saidaCamada) >= 2:
                        maxArgSaida = numpy.argmax(saidaCamada)
                        maxArgRotulo = numpy.argmax(rotulo)
                        if maxArgRotulo == maxArgSaida:
                            sum_values_accuracy[indexClasseSaida] += 1
                            arrAcertos[indexClasseSaida].append(1)
                        else:
                            arrAcertos[indexClasseSaida].append(0)
                    else:
                        throbleshot = 0.5
                        y = int(saidaCamada[0] >= throbleshot)
                        if y == rotulo[0]:
                            sum_values_accuracy[indexClasseSaida] += 1
                            arrAcertos[indexClasseSaida].append(1)
                        else:
                            arrAcertos[indexClasseSaida].append(0)

                acertos.append(arrAcertos)

            msgEntropy = "Entropy for camada "
            msgAcurracy = "Acurracy for camada "
            for indexCamadaSaida in range(len(self.arrCamadaSaida)):
                accuracy_for_camada[indexCamadaSaida] = sum_values_accuracy[indexCamadaSaida] / len(rotulos[i_previsoes])
                if isPrintar:
                    msgEntropy += f" [{indexCamadaSaida}: {entropy_for_camadas[indexCamadaSaida]:.5f}], "
                    msgAcurracy += f" [{indexCamadaSaida}: {accuracy_for_camada[indexCamadaSaida]:.5f}], "

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
            msgMediaEntropy += f" [{indexCamadaSaida}: {mediaEntropy[indexCamadaSaida]:.5f}], "
            msgMediaAcurracy += f" [{indexCamadaSaida}: {mediaAcurracy[indexCamadaSaida]:.5f}], "

        if isPrintar:
            print(msgMediaEntropy, "\n", msgMediaAcurracy)
            print("##########")

        return mediaEntropy, mediaAcurracy, acertos

    def prever(self, entradas: list[list], isPrintar: bool = True):
        pMatricialOcultos, ativacoesOcultas, pMatricialSaidas, arrSaidas = self.forward(entradas=entradas)
        print(arrSaidas)

        for indexEntrada in range(len(entradas)):
            saidaNorm = []
            for indexCamadaSaida in range(len(self.arrCamadaSaida)):
                saidaNorm.append([saida[indexEntrada] for saida in arrSaidas[indexCamadaSaida]])
            print(saidaNorm)


    def treinar(self, isBrekarPorEpocas=True, isAtualizarPesos=True, nEpocas=None, qtdeDadoValidar: int = 0,
                n_folds: int = 1):
        if qtdeDadoValidar >= 1:
            entradas_val = self.modelDataFF.arr_entradas[-qtdeDadoValidar:]
            rotulos_val = self.modelDataFF.arr_rotulos[-qtdeDadoValidar:]

            for i in range(qtdeDadoValidar):
                self.modelDataFF.arr_entradas.pop()
                self.modelDataFF.arr_rotulos.pop()
        else:
            entradas_val = self.modelDataFF.arr_entradas[-8:]
            rotulos_val = self.modelDataFF.arr_rotulos[-8:]

        arrArrSaidas_t = []
        arrArrSaidas_v = []

        arrDadosEntradas_t = []
        arrDadosEntradas_v = []

        arrDadosRotulos_t = []
        arrDadosRotulos_v = []

        lenEntrada = len(self.modelDataFF.arr_entradas)

        isValidarBom = False

        arrEntradas_nf = self.modelDataFF.iaRegras.obter_k_folds_temporal(self.modelDataFF.arr_entradas, n_folds=n_folds)
        arrRotulos_nf = self.modelDataFF.iaRegras.obter_k_folds_temporal(self.modelDataFF.arr_rotulos, n_folds=n_folds)

        if nEpocas is None:
            nEpocas = self.nEpocas

        index_fold_v = n_folds - 1
        isMaxPontuacao = False

        for i in range(nEpocas):
            for i_fold in range(n_folds):
                if i_fold != index_fold_v:
                    arrDadosEntradas_t = arrEntradas_nf[i_fold]
                    arrDadosRotulos_t = arrRotulos_nf[i_fold]

                    pMatricialOcultos, ativacoesOcultas_t, pMatricialSaidas, arrSaidas_t = self.forward(
                        entradas=arrDadosEntradas_t)
                    arrArrSaidas_t.append(arrSaidas_t)

                    if isAtualizarPesos:
                        self.backward(entradas=arrDadosEntradas_t, estados_ocultos=ativacoesOcultas_t, saidas_rede=arrSaidas_t,
                                      rotulos=arrDadosRotulos_t, pMatricalOcultas=pMatricialOcultos, pMatricialSaidas=pMatricialSaidas, acertos=[])

                elif i_fold == index_fold_v or n_folds == 1:
                    arrDadosEntradas_v = arrEntradas_nf[i_fold]


                    pMatricialOcultos, ativacoesOcultas_v, pMatricialSaidas, arrSaidas_v = self.forward(
                        entradas=arrDadosEntradas_v)

                    arrArrSaidas_v.append(arrSaidas_v)
                    arrDadosRotulos_v.append(arrRotulos_nf[i_fold])

                    if i % 10 == 0:
                        print(i, "de", nEpocas)
                        media_entripy, media_acurracy, acertos_v = self.calcular_erro(rotulos=arrDadosRotulos_v,
                                                                           arrPrevisoes=arrArrSaidas_v, isPrintar=True)
                        arrArrSaidas_v = []
                        arrDadosRotulos_v = []

                        if media_acurracy[-1] >= 1 and sum(media_acurracy) / len(media_acurracy) >= 0.93:
                            isValidarBom = True

                        if sum(media_acurracy) / len(media_acurracy) == 1.0:
                            isMaxPontuacao = True
                            #break

                    if isAtualizarPesos and n_folds == 1:
                        self.backward(entradas=arrDadosEntradas_v, estados_ocultos=ativacoesOcultas_v,
                                      saidas_rede=arrSaidas_v, rotulos=arrRotulos_nf[i_fold],
                                      pMatricalOcultas=pMatricialOcultos, pMatricialSaidas=pMatricialSaidas, acertos=[])

            index_fold_v = n_folds - 1 if index_fold_v == 0 else index_fold_v - 1

            saida_validar_v = self.forward(entradas=entradas_val)[3]
            self.media_entropy, self.media_accuracy, acertos_t = self.calcular_erro(rotulos=[rotulos_val], arrPrevisoes=[saida_validar_v], isPrintar=False)

            if (sum(self.media_accuracy) / len(self.media_accuracy) >= 1 and isValidarBom) or i % 100 == 0 or i == nEpocas - 1:
                print("Validação dados fora do dataset: ", i, " de ", nEpocas)
                self.calcular_erro(rotulos=[rotulos_val], arrPrevisoes=[saida_validar_v])
                if (sum(self.media_accuracy) / len(self.media_accuracy) >= 1 and isValidarBom) or (i == nEpocas - 1 and isValidarBom):
                    print("####### forward teste #######")
                    self.prever(entradas=entradas_val)
                    print("#############################")
                    break

            if isBrekarPorEpocas and i == nEpocas:
                break

        return self.media_entropy, self.media_accuracy
from __future__ import annotations

import math
import numpy

from matplotlib import pyplot
from api.regras.statisticsRegras import TeamInfoDataSet, TeamsPlaysEntrada, TeamsPlaysSaida

class IARegras:
    def normalizar_dataset(self, dataset, max_valor: list = None, min_valor: list = None) -> tuple[list, list, list]:
        dataset = numpy.asarray(dataset)
        max_valor = numpy.amax(dataset, axis=0) if max_valor is None else max_valor
        min_valor = numpy.amin(dataset, axis=0) if min_valor is None else min_valor

        arrDividendos = (max_valor - min_valor)
        dividendos = [dividendo if dividendo >= 1 else 1 for dividendo in arrDividendos]

        dataset_normalizado: list[list] = (dataset - min_valor) / arrDividendos
        return dataset_normalizado, max_valor, min_valor


    def obter_k_folds_temporal(self, arrDados: list, n_folds: int):
        len_k_folds = int(len(arrDados) / n_folds) + 1
        arrDadosCortados = list(arrDados)
        new_k_arr_dados = []
        new_k = []

        while len(arrDadosCortados) >= 1 :
            index = 0
            new_k.append(arrDadosCortados[index])
            arrDadosCortados.pop(index)

            if len(new_k) == len_k_folds:
                new_k_arr_dados.append(new_k)
                new_k = []

        if len(new_k) >= 1 and len(new_k_arr_dados) < n_folds:
            new_k_arr_dados.append(new_k)
        elif len(new_k) >= 1 and len(new_k_arr_dados) == n_folds:
            raise "Divisão dos dados errada"

        return new_k_arr_dados


    def normalizarDadosTeamInfoDataset(self, arrTeamsInfo: list[TeamInfoDataSet], arrIdsTeamPrever: list[int]) -> DatasetRNN:
        arrKeysIgnorar: list = ["data_fixture", "is_prever"]
        arrKeysEsperados: list = ["is_winner", "gols_marcados", "gols_sofridos"]
        #arrKeysEsperados: list = ["is_winner", "gols_marcados", "gols_sofridos", "is_winner_pen"]
        arrDadosEntrada: list = []
        arrDadosEsperados: list = []
        arrDadosPrever: list = []
        arrDadosEntadaFineTunnings: list = []
        arrDadosEsperadosFineTunnings: list = []

        for team in arrTeamsInfo:
            teamDict = team.__dict__
            newDadosEntradas = []
            newDadosEsperados = []

            for key in teamDict.keys():
                if key in arrKeysIgnorar:
                    continue
                elif key in arrKeysEsperados:
                    newDadosEsperados.append(teamDict[key])
                else:
                    newDadosEntradas.append(teamDict[key])

            if team.is_prever == 1:
                arrDadosPrever.append(newDadosEntradas)
            else:
                arrDadosEntrada.append(newDadosEntradas)
                arrDadosEsperados.append(newDadosEsperados)

                if newDadosEntradas[3] in arrIdsTeamPrever or newDadosEntradas[7] in arrIdsTeamPrever:
                    arrDadosEntadaFineTunnings.append(newDadosEntradas)
                    arrDadosEsperadosFineTunnings.append(newDadosEsperados)

        print(arrIdsTeamPrever)
        print(len(arrDadosEntrada), arrDadosEntrada[-1])
        print(len(arrDadosEsperados), arrDadosEsperados[-1])
        print(len(arrDadosPrever), arrDadosPrever[-1])

        arrDadosEntradaNormalizados, max_valor, min_valor = self.normalizar_dataset(dataset=arrDadosEntrada)
        arrDadosPreverNormalizados = self.normalizar_dataset(dataset=arrDadosPrever, max_valor=max_valor,
                                                             min_valor=min_valor)[0]

        arrDadosEsperadosNormalizados, max_esp, min_esp = self.normalizar_dataset(dataset=arrDadosEsperados)
        arrDadosEsperadosNormalizados = numpy.zeros((len(arrDadosEsperados), 15), dtype=numpy.int32).tolist()

        for indexDadosEsperados in range(len(arrDadosEsperados)):
            sumNewIndexDados = 0
            for indexDado in range(len(arrDadosEsperados[indexDadosEsperados])):
                valueIndex = arrDadosEsperados[indexDadosEsperados][indexDado]
                if sumNewIndexDados == 0:
                    arrDadosEsperadosNormalizados[indexDadosEsperados][valueIndex] = 1
                    sumNewIndexDados += 3
                else:
                    if valueIndex >= 5:
                        arrDadosEsperadosNormalizados[indexDadosEsperados][5 + sumNewIndexDados] = 1

                    else:
                        arrDadosEsperadosNormalizados[indexDadosEsperados][valueIndex + sumNewIndexDados] = 1

                    sumNewIndexDados += 5 + 1

        print(len(arrDadosEsperadosNormalizados), arrDadosEsperadosNormalizados[-1])

        arrDadosEntadaFineTunningsNormalizados = self.normalizar_dataset(dataset=arrDadosEntadaFineTunnings,
                                                                         max_valor=max_valor, min_valor=min_valor)[0]

        arrDadosEsperadosFineTunningsNormalizados = self.normalizar_dataset(dataset=arrDadosEsperadosFineTunnings,
                                                                            max_valor=max_esp, min_valor=min_esp)[0]
        arrDadosEsperadosFineTunningsNormalizados = numpy.zeros((len(arrDadosEsperadosFineTunnings), 15), dtype=numpy.int32).tolist()

        for indexDadosEsperados in range(len(arrDadosEsperadosFineTunnings)):
            sumNewIndexDados = 0
            for indexDado in range(len(arrDadosEsperadosFineTunnings[indexDadosEsperados])):
                valueIndex = arrDadosEsperadosFineTunnings[indexDadosEsperados][indexDado]
                if sumNewIndexDados == 0:
                    arrDadosEsperadosFineTunningsNormalizados[indexDadosEsperados][valueIndex] = 1
                    sumNewIndexDados += 3
                else:
                    if valueIndex >= 5:
                        arrDadosEsperadosFineTunningsNormalizados[indexDadosEsperados][5 + sumNewIndexDados] = 1
                    else:
                        arrDadosEsperadosFineTunningsNormalizados[indexDadosEsperados][valueIndex + sumNewIndexDados] = 1

                    sumNewIndexDados += 5 + 1

        newDatasetNormalizado = DatasetRNN(arr_entradas_treino=arrDadosEntradaNormalizados,
                                           arr_saidas_esperadas=arrDadosEsperadosNormalizados,
                                           arr_prevevisao=arrDadosPreverNormalizados)

        newDatasetNormalizado.arr_entradas_fine_treino = arrDadosEntadaFineTunningsNormalizados
        newDatasetNormalizado.arr_saidas_fine_esperadas = arrDadosEsperadosFineTunningsNormalizados

        newDatasetNormalizado.max_value_entradas = list(max_valor)
        newDatasetNormalizado.min_value_entradas = list(min_valor)
        newDatasetNormalizado.max_value_esperados = list(max_esp)
        newDatasetNormalizado.min_value_esperados = list(min_esp)
        newDatasetNormalizado.dado_exemplo = arrDadosPrever

        return newDatasetNormalizado


    def normalizarDadosTeamsPlayDataset(self, arrTeamsPlays: list[TeamsPlaysEntrada], arrIdsTeamPrever: list[int]) -> DatasetRNN:
        arrKeysIgnorar: list = ["data_fixture", "is_prever", "name_team_home", "name_team_away", "saida_prevista",
                                "qtde_gols_marcados_away", "qtde_gols_marcados_home"]
        arrDadosEntrada: list = []
        arrDadosEsperados: list = []
        arrDadosPrever: list = []

        ordemNameValuesEntrada: list[str] = []
        ordemNameValuesSaida: list[str] = []

        isSalvouOrdem = False

        for team in arrTeamsPlays:
            if team.id_team_away not in arrIdsTeamPrever and team.id_team_home not in arrIdsTeamPrever:
                continue

            teamSaidaDict = team.saida_prevista.__dict__
            teamEntradaDict = team.__dict__

            newDadosEntradas = []
            newDadosEsperados = []

            for key in teamEntradaDict.keys():
                if key in arrKeysIgnorar:
                    continue
                else:
                    if not isSalvouOrdem:
                        ordemNameValuesEntrada.append(key)
                    newDadosEntradas.append(teamEntradaDict[key])

            for key in teamSaidaDict:
                if not isSalvouOrdem:
                    ordemNameValuesSaida.append(key)
                newDadosEsperados.append(teamSaidaDict[key])

            isSalvouOrdem = True

            if team.is_prever == 1:
                arrDadosPrever.append(newDadosEntradas)
            else:
                arrDadosEntrada.append(newDadosEntradas)
                arrDadosEsperados.append(newDadosEsperados)


        arrDadosEntradaNormalizados, max_valor, min_valor = self.normalizar_dataset(dataset=arrDadosEntrada)
        arrDadosPreverNormalizados = self.normalizar_dataset(dataset=arrDadosPrever, max_valor=max_valor,
                                                             min_valor=min_valor)[0]

        arrDadosEsperadosNormalizados, max_esp, min_esp = self.normalizar_dataset(dataset=arrDadosEsperados)

        arrDadosEsperadosNormalizadosEmClasses = []

        for dadoEsperados in arrDadosEsperados:
            arrDadosClasse = []
            for index_max_value in range(len(max_esp.tolist())):
                max_value = max_esp[index_max_value]
                dado_value = dadoEsperados[index_max_value]

                for i in range(max_value + 1):
                    classe = 0
                    if dado_value == i:
                        classe = 1

                    arrDadosClasse.append(classe)

            arrDadosEsperadosNormalizadosEmClasses.append(arrDadosClasse)

        newDatasetNormalizado = DatasetRNN(arr_entradas_treino=arrDadosEntradaNormalizados,
                                           arr_saidas_esperadas=arrDadosEsperadosNormalizadosEmClasses,
                                           arr_prevevisao=arrDadosPreverNormalizados)

        newDatasetNormalizado.max_value_entradas = list(max_valor)
        newDatasetNormalizado.min_value_entradas = list(min_valor)
        newDatasetNormalizado.max_value_esperados = list(max_esp)
        newDatasetNormalizado.min_value_esperados = list(min_esp)
        newDatasetNormalizado.arr_name_values_entrada = ordemNameValuesEntrada
        newDatasetNormalizado.arr_name_values_saida = ordemNameValuesSaida
        newDatasetNormalizado.dado_exemplo = arrDadosPrever

        return newDatasetNormalizado

class DatasetRNN:
    def __init__(self, arr_entradas_treino: list, arr_saidas_esperadas: list, arr_prevevisao: list):
        self.arr_entradas_treino: list = arr_entradas_treino
        self.arr_saidas_esperadas: list = arr_saidas_esperadas
        self.arr_prevevisao: list = arr_prevevisao
        self.arr_entradas_treino_DBN: list = []
        self.arr_prevevisao_DBN: list = []

        self.arr_entradas_fine_treino: list = []
        self.arr_saidas_fine_esperadas: list = []

        self.max_value_entradas: list = []
        self.min_value_entradas: list = []

        self.max_value_esperados: list = []
        self.min_value_esperados: list = []

        self.arr_name_values_entrada: list[str] = []
        self.arr_name_values_saida: list[str] = []

        self.dado_exemplo: any = None


        if len(arr_saidas_esperadas) != len(arr_saidas_esperadas):
            raise "Datasets incompletos"

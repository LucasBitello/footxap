from __future__ import annotations
import sys, threading
import time
from datetime import  datetime
from api.regras.apiFootball import RegraAPIFootBall

import mysql.connector as connection

class Database:
    def __init__(self):
        self.conexao = None

    def openConnection(self):
        isConectou = False
        while not isConectou:
            try:
                if not self.conexao:
                    self.conexao = connection.connect(database="footxap", user="root", password="pudinzinho", host="127.0.0.1")
                isConectou = True
            except connection.Error as cExc:
                time.sleep(2)
                print("abrir conexao falhou tentando novamente.")
                print("Ocorreu a exceção", cExc)
                isConectou = False

    def closeConnection(self):
        isFechou = False
        while not isFechou:
            try:
                #self.conexao.close()
                isFechou = True
            except connection.Error as cExc:
                time.sleep(2)
                print("fechar conexao falhou tentando novamente.")
                print("Ocorreu a exceção", cExc)
                isFechou = False


    def commit(self):
        self.conexao.commit()

    def rollback(self):
        self.conexao.rollback()

    def startTransaction(self):
        self.conexao.start_transaction()

    def executeSelectQuery(self, query: str, classModelDB: any, params: list = []) -> list[classModelDB]:
        self.openConnection()
        with self.conexao.cursor() as cursor:
            cursor.execute(query, params)
            nameColumns = [col[0] for col in cursor.description]
            rowsData = cursor.fetchall()
            data = [classModelDB(dict(zip(nameColumns, row))) for row in rowsData]

            self.closeConnection()
            return data


    def executeModifyQuery(self, query: str, params: list = [], isCommit: bool = True) -> list[int, int]:
        self.openConnection()
        with self.conexao.cursor() as cursor:
            cursor.execute(query, params)

            if isCommit:
                self.commit()

            self.closeConnection()
            return [cursor.lastrowid, cursor.rowcount]

class ClassModel:
    def __init__(self, dado: dict|object):
        if dado is not None:
            if type(dado) != dict:
                dado = dado.__dict__

            for key in dado.keys():
                self.__setattr__(key, dado[key])

    def getDict(self):
        return self.__dict__


class AtualizarDatabase:

    def __init__(self):
        self.database = Database()

    def atualizarTabela(self, model: Model, functionAtualizacao: any, isForçarAtualização: bool = False):
        self.criarTabelaTableSettings()

        isAtualizarTabela: bool = False
        querySelectTable = f"SELECT * FROM table_settings WHERE id_table = {model.id_tabela}"
        arrTableSettings: list[self.TableSettings] = self.database.executeSelectQuery(query=querySelectTable,
                                                                                      classModelDB=self.TableSettings)

        if len(arrTableSettings) >= 2:
            raise f"Table settings com registros duplicados id_table: {model.id_tabela}"

        elif len(arrTableSettings) == 0 and model.rateRefeshTableInMs >= 1:
            queryInsertNewTable = f"INSERT INTO table_settings (id_table, name_table, rate_refresh_in_ms)" \
                                  f" VALUES ({model.id_tabela}, '{model.name_table}', {model.rateRefeshTableInMs})"

            self.database.executeModifyQuery(query=queryInsertNewTable, isCommit=True)
            isAtualizarTabela = True

        elif len(arrTableSettings) == 1:
            tableSettings = arrTableSettings[0]
            dateUltimaAtualizacao = tableSettings.last_modification

            if dateUltimaAtualizacao is None:
                isAtualizarTabela = True
            else:
                if tableSettings.rate_refresh_in_ms != model.rateRefeshTableInMs:
                    queryUpdateRateRefresh = f"UPDATE table_settings SET rate_refresh_in_ms = {model.rateRefeshTableInMs}" \
                                             f" WHERE id_table = {model.id_tabela}"

                    self.database.executeModifyQuery(query=queryUpdateRateRefresh, isCommit=True)
                    tableSettings.rate_refresh_in_ms = model.rateRefeshTableInMs

                if type(dateUltimaAtualizacao) != datetime:
                    dateUltimaAtualizacaoNormalizada = datetime.strptime(dateUltimaAtualizacao, "%Y-%m-%d %H:%M:%S")
                else:
                    dateUltimaAtualizacaoNormalizada = dateUltimaAtualizacao

                dateNow = datetime.now()
                diff_ms = int((dateNow - dateUltimaAtualizacaoNormalizada).total_seconds() * 1000)

                if diff_ms >= tableSettings.rate_refresh_in_ms:
                    isAtualizarTabela = True

        if isAtualizarTabela or isForçarAtualização:
            functionAtualizacao()
            self.atualizarTableSettingsLastModification(model.id_tabela)

    def atualizarTableSettingsLastModification(self, id_table: int):
        dateNow = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = f"UPDATE table_settings SET last_modification = '{dateNow}' WHERE id_table = {id_table}"
        self.database.executeModifyQuery(query=query, isCommit=True)


    def criarTabelaTableSettings(self):
        query = """
            CREATE TABLE IF NOT EXISTS `table_settings` (
                `id` INT NOT NULL AUTO_INCREMENT,
                `id_table` INT NOT NULL,
                `name_table` VARCHAR(45) NOT NULL,
                `rate_refresh_in_ms` BIGINT UNSIGNED NOT NULL DEFAULT 0,
                `last_modification` DATETIME NULL,
                PRIMARY KEY (`id`),
                UNIQUE (`id_table`));
        """
        self.database.executeModifyQuery(query=query, params=[])

    class TableSettings(ClassModel):
        def __init__(self, tableSettings: dict | object = None):
            self.id: int = None
            self.id_table: int = None
            self.name_table: str = None
            self.rate_refresh_in_ms: int = None
            self.last_modification: str = None

            super().__init__(tableSettings)


class Model(AtualizarDatabase):
    name_db = "footxap"

    def __init__(self, name_table: str, id_tabela: int, name_columns_id: list[str],
                 reference_db_api: list[ReferenciaDatabaseToAPI], referencia_tabelas_filhas: list[ReferenciaTabelasFilhas],
                 referencia_tabelas_pai: list[ReferenciaTabelasPai], classModelDB: object, rate_refesh_table_in_ms: int) -> None:

        self.id_tabela = id_tabela
        self.name_table = name_table
        self.columns_id = name_columns_id
        self.classModelDB = classModelDB
        self.reference_db_api = reference_db_api
        self.referencia_tabelas_filhas = referencia_tabelas_filhas
        self.is_tabela_pai = len(referencia_tabelas_filhas) >= 1
        self.referencia_tabelas_pai = referencia_tabelas_pai
        self.is_tabela_filha = len(referencia_tabelas_pai) >= 1
        self.classIdsTable = IdTabelas()
        self.rateRefeshTableInMs = rate_refesh_table_in_ms

        self.name_column_last_modification = "last_modification"
        self.formato_datetime_YYYY_MM_DD_H_M_S = "%Y-%m-%d %H:%M:%S"

        self.database = Database()
        self.atualizarDatabase = super()
        self.regraApiFootBall = RegraAPIFootBall()


    def insert(self, data: dict) -> int:
        name_table = self.name_table
        dataInsert = []

        if type(data) == dict:
            dataInsert = data
        else:
            raise "Insert está errado, os dados não estão no formato de dict."

        arrInterrogation = []
        arrData = []
        arrkeys = []

        if (self.name_column_last_modification in dataInsert.keys()) and dataInsert[self.name_column_last_modification] is None:
            dataInsert[self.name_column_last_modification] = datetime.now().strftime(self.formato_datetime_YYYY_MM_DD_H_M_S)

        for key in dataInsert.keys():
            if dataInsert[key] is None:
                continue

            arrkeys.append(key)
            arrInterrogation.append('%s')
            arrData.append(dataInsert[key])

        query = "INSERT INTO " + name_table + "("
        query += ",".join(arrkeys) + ") values ("
        query += ",".join(arrInterrogation) + ")"

        lastIdInserted = self.database.executeModifyQuery(query, arrData)[0]
        return lastIdInserted


    def update(self, data: dict) -> int:
        name_table = self.name_table
        columns_id = self.columns_id
        dataUpdate = []

        if type(data) == dict:
            dataUpdate = data
        else:
            raise "Update está errado, os dados não estão no formato correto de dict."

        arrData = []
        arrStrUpdateColumn = []

        for key in dataUpdate.keys():
            if dataUpdate[key] is None:
                continue

            arrData.append(dataUpdate[key])
            strUpdateColumn = "" + key + " = %s"
            arrStrUpdateColumn.append(strUpdateColumn)

        query = "UPDATE " + name_table + " SET " + ",".join(arrStrUpdateColumn)

        queryWhere = " WHERE "
        arrDataWhere = []
        arrQuerysWhere = []

        for index_name_column_id in range(len(columns_id)):
            name_column_id = columns_id[index_name_column_id]
            arrQuerysWhere.append("" + name_column_id + " = %s")
            arrDataWhere.append(dataUpdate[name_column_id])

        queryWhere += " AND ".join(arrQuerysWhere) + ";"

        query += queryWhere
        arrData = arrData + arrDataWhere
        infoReturnDB = self.database.executeModifyQuery(query, arrData)
        qtdeRowsAfetadas = infoReturnDB[1]

        """if qtdeRowsAfetadas >= 1:
            datetimeNow = datetime.now().strftime(self.formato_datetime_YYYY_MM_DD_H_M_S)
            queryAttLastUpdated = "UPDATE " + self.name_table \
                                  + " SET " + self.name_column_last_modification  + " = '" + datetimeNow+"'" \
                                  + queryWhere
            self.database.executeModifyQuery(queryAttLastUpdated, arrDataWhere)"""

        lastId = dataUpdate[columns_id[0]]
        return lastId

    def delete(self):
        pass

    def executarQuery(self, query: str, params: list):
        self.database.executeModifyQuery(query=query, params=params)

    def salvar(self, data: list[dict], isNormalizarDados: bool = True, isForcarInsert: bool = False,
               isAtualizarUltimoUpdate: bool = False, isCommitar: bool = True) -> IdsModificadosDatabase:
        arrDataUpdate = data

        if isNormalizarDados:
            arrDataUpdate = self.normalizarDados(self.classModelDB, data)

        if type(arrDataUpdate) != list:
            arrDataUpdate = [data]
        else:
            if type(arrDataUpdate[0]) != dict:
                raise "Algo errado ao salvar"

        classIdsModificados = IdsModificadosDatabase()

        for dataUpdate in arrDataUpdate:
            isAllIdsPreenchidos = True

            dadosColunsId = []

            for name_column_id in self.columns_id:
                if dataUpdate[name_column_id] is None:
                    isAllIdsPreenchidos = False
                else:
                    dadosColunsId.append(dataUpdate[name_column_id])

            if isAllIdsPreenchidos and not isForcarInsert:
                arrDadosExistentes = self.obterByColumnsID(dadosColunsId)
                if len(arrDadosExistentes) == 0:
                    id_modify = self.insert(dataUpdate)
                    classIdsModificados.insert.append(id_modify)
                else:
                    id_modify = self.update(dataUpdate)
                    classIdsModificados.update.append(id_modify)
            else:
                id_modify = self.insert(dataUpdate)
                classIdsModificados.insert.append(id_modify)

        return classIdsModificados

    def normalizarDados(self, classTable, data: list) -> list[dict]:
        arrDataUpdate = data
        arrDadosNormalizados = []

        if type(data) != list:
            arrDataUpdate = [data]

        for dataUpdate in arrDataUpdate:
            newDataUpdate = classTable(dataUpdate).getDict()
            arrDadosNormalizados.append(newDataUpdate)

        return arrDadosNormalizados


    def obterTudo(self) -> list[self.classModelDB]:
        query = "SELECT * FROM " + self.name_table
        arrDados = self.database.executeSelectQuery(query=query, classModelDB=self.classModelDB)
        return arrDados

    def obterByReferenceApi(self, dadosBusca: list[str|int], isNoneIfNotResults: bool = False) -> list[self.classModelDB]:
        arrStrConsulta: list[str] = []

        if len(dadosBusca) != len(self.reference_db_api):
            raise "Busca por dados de referencia incompletos"

        for referencia in self.reference_db_api:
            arrStrConsulta.append(" " + referencia.nome_coluna_db+ " = (%s)")


        query = "SELECT * FROM " + self.name_table + " WHERE " + " AND ".join(arrStrConsulta)
        dados = self.database.executeSelectQuery(query=query, classModelDB=self.classModelDB, params=dadosBusca)

        if isNoneIfNotResults and len(dados) == 0:
            return None

        return dados

    def obterIdByReferenceIdApi(self, idApi: int) -> int|None:
        arrDados = self.obterByReferenceApi([idApi])

        if len(arrDados) == 0:
            return None
        else:
            if len(self.columns_id) >= 2:
                raise "Muitos parametros para retornar um unico id"

            return arrDados[0].getDict()[self.columns_id[0]]


    def obterByColumns(self, arrNameColuns: list[str], arrDados: list, clausulaOrder: str = None) -> list[self.classModelDB]:
        if len(arrNameColuns) != len(arrDados):
            raise "Porra meu quantia de parametros errados na consulta do método base: obterByColumns"

        query = "SELECT * FROM " + self.name_table + " where"
        arrQueryWhere = []

        for index in range(len(arrNameColuns)):
            arrQueryWhere.append(" " + arrNameColuns[index] + " = (%s)")

        query += " AND ".join(arrQueryWhere)

        if clausulaOrder is not None:
            query += " " + clausulaOrder

        arrRetorno = self.database.executeSelectQuery(query=query, classModelDB=self.classModelDB, params=arrDados)
        return arrRetorno


    def obterByColumnsID(self, arrDados: list) -> list[self.classModelDB]:
        if len(self.columns_id) != len(arrDados):
            raise "Porra meu quantia de parametros errados na consulta do método base: obterByColumnsID"

        query = "SELECT * FROM " + self.name_table + " where"
        arrQueryWhere = []

        for index in range(len(self.columns_id)):
            arrQueryWhere.append(" " + self.columns_id[index] + " = (%s)")

        query += " AND ".join(arrQueryWhere)

        arrRetorno = self.database.executeSelectQuery(query=query, classModelDB=self.classModelDB, params=arrDados)
        return arrRetorno


    def normalizarDadosForView(self, arrDados: list[self.classModelDB]) -> list[dict]:
        arrNormalizado: list[dict] = [dado.getDict() for dado in arrDados]
        return arrNormalizado


class IdsModificadosDatabase:
    def __init__(self):
        self.insert: list = []
        self.update: list = []

    def getID(self) -> int:
        if len(self.insert) >= 1:
            return self.insert[0]
        elif len(self.update) >= 1:
            return self.update[0]
        else:
            return None

class ReferenciaDatabaseToAPI:
    def __init__(self, nome_coluna_db, nome_coluna_api):
        self.nome_coluna_db: str = nome_coluna_db
        self.nome_coluna_api: str = nome_coluna_api


class ReferenciaTabelasFilhas:
    def __init__(self, id_tabela_filha: int, nome_tabela_filha: str, nome_coluna_tabela_pai: str, nome_coluna_tabela_filha: str):
        self.id_tabela_filha: int = id_tabela_filha
        self.nome_tabela_filha: str = nome_tabela_filha
        self.nome_coluna_tabela_pai: str = nome_coluna_tabela_pai
        self.nome_coluna_tabela_filha: str = nome_coluna_tabela_filha

class ReferenciaTabelasPai:
    def __init__(self, id_tabela_pai: int, nome_tabela_pai: str, nome_coluna_tabela_pai: str, nome_coluna_tabela_filha: str):
        self.id_tabela_pai: int = id_tabela_pai
        self.nome_tabela_pai: str = nome_tabela_pai
        self.nome_coluna_tabela_pai: str = nome_coluna_tabela_pai
        self.nome_coluna_tabela_filha: str = nome_coluna_tabela_filha


class IdTabelas:
    def __init__(self):
        self.table_settings = 1
        self.country = 2
        self.league = 3
        self.season = 4
        self.team = 5
        self.team_venue = 6
        self.team_seasons = 7
        self.fixture = 8
        self.fixture_teams = 9
        self.type_fixture_team_statistic = 10
        self.fixture_team_estatistics = 11
        self.type_fixture_team_lineup = 12
        self.fixture_team_lineups = 13



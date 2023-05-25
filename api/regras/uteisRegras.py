import datetime
from api.models.model import Database

class ParamsNotNone:
    def __init__(self):
        self.nameColumns: list[str] = []
        self.dataColumns: list = []

class UteisRegras():
    def retornarSomenteParamsNotNone(self, dictDados: dict) -> ParamsNotNone:
        paramsNotNone: ParamsNotNone = ParamsNotNone()
        for key in dictDados.keys():
            if dictDados[key] is not None:
                paramsNotNone.nameColumns.append(key)
                paramsNotNone.dataColumns.append(dictDados[key])

        return paramsNotNone

    def normalizarDadosForView(self, arrDados: list[object], isFecharConexao: bool = True) -> list[dict]:
        database = Database()
        arrDadosJson = []
        arrDadosNormalizados = []

        for dado in arrDados:
            if (type(dado) != int and type(dado) != float and type(dado) != str and type(dado) != list and dado is not None):
                if type(dado) != dict:
                    arrDadosJson.append(dado.__dict__)
                else:
                    arrDadosJson.append(dado)
            else:
                arrDadosNormalizados.append(dado)

        for dado in arrDadosJson:
            for key in dado.keys():
                if type(dado[key]) == list:
                    dado[key] = self.normalizarDadosForView(dado[key])
                elif (type(dado[key]) != int and type(dado[key]) != float and type(dado[key]) != str and
                      type(dado[key]) != list and dado[key] is not None and type(dado[key]) != datetime.datetime and
                      type(dado[key]) != datetime.date and type(dado[key]) != dict and type(dado[key]) != bool):
                    dado[key] = dado[key].__dict__
                    for key2 in dado[key]:
                        if type(dado[key][key2]) == list and len(dado[key][key2]) >= 1:
                            dado[key][key2] = self.normalizarDadosForView(dado[key][key2])
                        elif (type(dado[key][key2]) != int and type(dado[key][key2]) != float and
                              type(dado[key][key2]) != str and type(dado[key][key2]) != list and
                              dado[key][key2] is not None and type(dado[key][key2]) != datetime.datetime and
                              type(dado[key][key2]) != datetime.date and type(dado[key][key2]) != dict):
                            dado[key][key2] = dado[key][key2].__dict__

            arrDadosNormalizados.append(dado)

        database.closeConnection()
        return arrDadosNormalizados
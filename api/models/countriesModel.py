from __future__ import annotations
from api.models.model import Model, IdTabelas, ReferenciaDatabaseToAPI, ReferenciaTabelasFilhas, ReferenciaTabelasPai, ClassModel

class CountriesModel(Model):
    def __init__(self):
        super().__init__(
            name_table="country",
            id_tabela=IdTabelas().country,
            name_columns_id=["id"],
            reference_db_api=[ReferenciaDatabaseToAPI(nome_coluna_db="name", nome_coluna_api="name")],
            referencia_tabelas_filhas=[ReferenciaTabelasFilhas(id_tabela_filha=IdTabelas().league,
                                                               nome_tabela_filha="league",
                                                               nome_coluna_tabela_pai="id",
                                                               nome_coluna_tabela_filha="id_country",
                                                               ),
                                       ReferenciaTabelasFilhas(id_tabela_filha=IdTabelas().team,
                                                               nome_tabela_filha="team",
                                                               nome_coluna_tabela_pai="id",
                                                               nome_coluna_tabela_filha="id_country")],
            referencia_tabelas_pai=[],
            classModelDB=Country,
            rate_refesh_table_in_ms=31536000000)

        self.criarTableDataBase()


    def criarTableDataBase(self):
        query = f"""CREATE TABLE IF NOT EXISTS {self.name_table}  (
                  `id` INT NOT NULL AUTO_INCREMENT,
                  `name` VARCHAR(45) NOT NULL,
                  `code` VARCHAR(45) NULL,
                  `flag` LONGTEXT NULL,
                  `is_obter_dados` TINYINT NOT NULL  DEFAULT 0,
                  `last_modification` DATETIME NOT NULL,
                  PRIMARY KEY (`id`),
                  UNIQUE (`name`));"""

        self.executarQuery(query=query, params=[])


    def obterNamebyId(self, id: int) -> str:
        arrContries: list[Country] = self.obterByColumnsID(arrDados=[id])

        if len(arrContries) == 0:
            return None
        else:
            return arrContries[0].name


    def fazerConsultaApiFootball(self, name: str = None, code: str = None, search: str = None) -> list[dict]:
        arrParams = []
        query = "countries"
        nameColumnResponseData = "response"


        if name is not None:
            arrParams.append("name=" + name)
        if code is not None:
            arrParams.append("code=" + code)
        if search is not None:
            arrParams.append("search=" + search)

        if len(arrParams) >= 1:
            query += "?" + "&".join(arrParams)

        response = self.regraApiFootBall.conecarAPIFootball(query)
        responseData = response[nameColumnResponseData]

        return responseData


    def atualizarDBCountries(self, name_country: str = None) -> None:
        arrCountries = self.fazerConsultaApiFootball(name=name_country)

        for data in arrCountries:
            if data["code"] is None:
                data["flag"] = "https://img.freepik.com/vetores-gratis/terra-isolada-em-branco_1308-55360.jpg"

            country: Country = Country()
            country.name = data["name"]
            country.code = data["code"]
            country.flag = data["flag"]
            country.id = self.obterIdByReferenceIdApi(idApi=data["name"])
            country.is_obter_dados = 1 if country.name == "World" else 0

            self.salvar(data=[country])

    def atualizarDados(self):
        functionAttDB = lambda: self.atualizarDBCountries()
        arrCountries: list[Country] = self.obterTudo()
        isForçarAtualização = len(arrCountries) == 0
        self.atualizarTabela(model=self, functionAtualizacao=functionAttDB, isForçarAtualização=isForçarAtualização)

class Country(ClassModel):
    def __init__(self, country: dict|object = None):
        self.id: int = None
        self.name: str = None
        self.code: str = None
        self.flag: str = None
        self.is_obter_dados: int = None
        self.last_modification: str = None

        super().__init__(country)
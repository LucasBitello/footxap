from __future__ import annotations
from api.models.model import Model, ReferenciaDatabaseToAPI, ReferenciaTabelasFilhas, IdTabelas, ClassModel, ReferenciaTabelasPai

class TeamsVenueModel(Model):
    def __init__(self):
        super().__init__(
            name_table="team_venue",
            id_tabela=IdTabelas().team_venue,
            name_columns_id=["id"],
            reference_db_api=[ReferenciaDatabaseToAPI(nome_coluna_db="id_api", nome_coluna_api="id")],
            referencia_tabelas_pai=[ReferenciaTabelasPai(id_tabela_pai=IdTabelas().team,
                                                         nome_tabela_pai="team",
                                                         nome_coluna_tabela_pai="id",
                                                         nome_coluna_tabela_filha="id_team")],
            referencia_tabelas_filhas=[],
            classModelDB=TeamVenue,
            rate_refesh_table_in_ms=0)

        self.criarTableDataBase()


    def criarTableDataBase(self):
        query = f"""CREATE TABLE IF NOT EXISTS {self.name_table} (
            `id` INT NOT NULL AUTO_INCREMENT,
            `id_api` INT NOT NULL,
            `id_team` INT NOT NULL,
            `name` VARCHAR(255) NULL,
            `address` MEDIUMTEXT NULL,
            `city` VARCHAR(255) NULL,
            `capacity` INT NULL,
            `surface` VARCHAR(45) NULL,
            `image` MEDIUMTEXT NULL,
            `last_modification` DATETIME NOT NULL,
            PRIMARY KEY (`id`),
            CONSTRAINT `id_team_vte_tea`
            FOREIGN KEY (`id_team`)
            REFERENCES `team` (`id`)
            ON DELETE RESTRICT
            ON UPDATE RESTRICT,
            UNIQUE (`id_api`));"""

        self.executarQuery(query=query, params=[])

    def atualizarDBTeamVenue(self, teamVenue: dict, id_team_db: int) -> None:
        id_venue_salvo: TeamVenue = self.obterIdByReferenceIdApi(teamVenue["id"])

        newTeamVenue = TeamVenue()
        newTeamVenue.id = id_venue_salvo
        newTeamVenue.id_api = teamVenue["id"]
        newTeamVenue.id_team = id_team_db
        newTeamVenue.name = teamVenue["name"]
        newTeamVenue.address = teamVenue["address"]
        newTeamVenue.city = teamVenue["city"]
        newTeamVenue.capacity = teamVenue["capacity"]
        newTeamVenue.surface = teamVenue["surface"]
        newTeamVenue.image = teamVenue["image"]

        self.salvar(data=[newTeamVenue])

class TeamVenue(ClassModel):
    def __init__(self, teamVenue: dict|object = None):
        self.id: int = None
        self.id_api: int = None
        self.id_team: int = None

        self.name: str = None
        self.address: str = None
        self.city: str = None
        self.capacity: int = None
        self.surface: str = None
        self.image: str = None
        self.last_modification: str = None

        super().__init__(dado=teamVenue)
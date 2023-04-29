from api.models.teamsModel import TeamsModel, Team

class TeamsRegras:
    def __init__(self):
        self.teamsModel = TeamsModel()

    def obter(self, id: int = None, name: str = None, id_season: int = None):
        if id is not None and id_season is None:
            self.teamsModel.atualizarDados(id_team=id)
            arrTeams = self.teamsModel.obterByColumnsID(arrDados=[id])
        elif id_season is not None:
            if name is None:
                name = ""

            arrTeams = self.teamsModel.obterTeamsBySeasonName(name=name, id_season=id_season, id_team=id)
        elif name is not None:
            arrTeams = self.teamsModel.obterTeamsByName(name_team=name)
        else:
            raise "nenhum paramentro válido"

        return arrTeams
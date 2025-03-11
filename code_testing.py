# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests",
# ]
# ///
from dataclasses import dataclass
from pprint import pprint

import requests
from requests.exceptions import RequestException


@dataclass
class Bet:
    driver_id: int
    credits: int

    def serialize(self):
        """Serialize the dataclass for a request"""
        return {'driverId': self.driver_id, 'credits': self.credits}


@dataclass
class Driver:
    id: int
    name: str
    years_of_experience: int
    driving_style: str


def get_current_credits(api_url: str, team_id: str, log: bool = False) -> int:
    """Get the amount of credits your team currently has"""

    response = requests.get(f'{api_url}/teams/{team_id}')

    if response.status_code != 200:
        raise RequestException(f'Error fetching team data: {response.json()["detail"]}')

    team_credits = response.json()['credits']

    if log:
        print(f'Your team has {team_credits} credits')
        print()

    return team_credits


def get_next_race(api_url: str, log: bool = False) -> int:
    """Get all information for the next race, return the id"""

    response = requests.get(f'{api_url}/races/next')

    if response.status_code != 200:
        raise RequestException(f'Error fetching race data: {response.json()["detail"]}')

    response_json = response.json()

    if log:
        print('Retrieved the following data for the next race:')
        pprint(response_json)
        print()

    return response_json['id']


def get_drivers(api_url: str, log: bool = False) -> list[Driver]:
    """Get all driver information"""

    response = requests.get(f'{api_url}/drivers')

    if response.status_code != 200:
        raise RequestException(
            f'Error getting driver information: {response.json()["detail"]}'
        )

    # parse the returned json into driver dataclasses
    drivers = [
        Driver(
            driver_json['id'],
            driver_json['name'],
            driver_json['yearsOfExperience'],
            driver_json['drivingStyle'],
        )
        for driver_json in response.json()
    ]

    if log:
        print('Retrieved the following driver data:')
        pprint(drivers)
        print()

    return drivers


def place_example_bets(
    api_url: str, team_id: str, race_id: int, log: bool = False
) -> None:
    """Places bets for the next race"""

    # for this example, let's place bets on the 3 drivers with the most experience
    drivers = get_drivers(api_url)
    best_drivers = sorted(
        drivers, key=lambda driver: driver.years_of_experience, reverse=True
    )[:3]

    # place a bet for each driver using a fraction of our credits
    current_credits = get_current_credits(api_url, team_id)
    example_bets = {
        'bets': [
            Bet(driver.id, current_credits // 5).serialize() for driver in best_drivers
        ]
    }

    # make the put request to place the bets
    response = requests.put(f'{api_url}/bets/{team_id}/{race_id}', json=example_bets)

    # if the request was not valid, raise an exception with the error
    if response.status_code != 200:
        raise RequestException(f'Error placing bets: {response.json()["detail"]}')

    # if the request was valid, it will return the placed bets
    if log:
        print('Successfully placed the following bets:')
        pprint(response.json())
        print()


def main():
    api_url = 'https://rdwvirtualracing.azurewebsites.net'

    secret_team_id = 'enter code from the paper here'

    next_race_id = get_next_race(api_url, log=True)

    place_example_bets(api_url, secret_team_id, next_race_id, log=True)


if __name__ == '__main__':
    main()

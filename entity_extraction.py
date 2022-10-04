import pandas as pd
import plotly.express as px
import spacy
from collections import Counter
nlp = spacy.load('en_core_web_md')
import chart_studio.plotly as py
import chart_studio

london_locations = {
    'Whitehall': (51.5051758449122, -0.1265497441844789),
    'White hall': (51.5051758449122, -0.1265497441844789),
    'Westminster': (51.499903812847826, -0.12721346931036084),
    'Parliament': (51.499903812847826, -0.12721346931036084),
    'Westminster Hall': (51.499903812847826, -0.12721346931036084),
    'the House of Lords': (51.499903812847826, -0.12721346931036084),
    'Deptford': (51.48154560328973, -0.023094367821352133),
    'Woolwich': (51.48957087033001, 0.06586413481564578),
    'Cheapside': (51.51437617547665, -0.09459917302028253),
    "Charing Cross": (51.50921552379641, -0.12539875313544238),
    "Ludgate Hill": (51.51414369232536, -0.10247055767546208),
    "Fleet Street": (51.5143978753939, -0.10797433069269882),
    "Greenwich": (51.493434630287055, 0.009036046085412489),
    "Islington": (51.53805133403498, -0.10429763894136518),
    'Hampton Court': (51.40465671755744, -0.33598699226306955),
    'Covent Garden': (51.51185572404155, -0.12445607895330543),
    'Lumbard Street': (51.512753811571855, -0.08785943388228405),
    'Lombard Street': (51.512753811571855, -0.08785943388228405),
    'Lambeth': (51.49398155154983, -0.11854751904932402),
    'Hercules Pillars': (51.51613445241099, -0.1202303716841065),
    'Chatham': (51.39834455619583, 0.5220144944578242),

}

def get_counts_of_spec_entity(entities, entity_type):

    entity_counts = []
    for row in entities:
        for ent in row.ents:
            if ent.label_ == entity_type:
                entity_counts.append(ent.text)
    return entity_counts


def get_counts_of_all_entity(entities, top_n):

    entity_counts = []
    for row in entities:
        for ent in row.ents:
            entity_counts.append(ent.text)
    return Counter(entity_counts).most_common(top_n)

def get_entity_type_counts(entities):

    counts = []
    for row in entities:
        for ent in row.ents:
            counts.append(ent.label_)
    return Counter(counts)


def plot_locations(df, zoom=10, radius=30, opacity=1, mapbox_style="carto-positron", range_color=None):

    # TO add chart to Plotly uncomment and fill in the following with plotly account details:
    username = 'xxx'
    api_key = 'xxx'
    # chart_studio.tools.set_credentials_file(username=username,
    #                                         api_key=api_key)
    fig = px.density_mapbox(df, lat='latitude', lon='longitude', z='count', radius=radius,
                            center=dict(lat=51.499903812847826, lon=-0.12721346931036084), zoom=zoom, opacity=opacity,
                            mapbox_style=mapbox_style,
                            hover_data=['location_name'],
                            range_color=range_color
                            )
    fig.show()
    # unccoment the following to add to plotly
    # py.plot(fig, filename="plotly_scatter", auto_open=True)


def extract_entities(df):
    # replace usage of 'streete' with 'street' as Pepys uses these inconsistently
    df['entry'] = df['entry'].str.replace('Streete', 'Street')
    # df['processed'] = df['entry'].apply(sent_tokenize)
    df['entities'] = df['entry'].apply(nlp)
    entity_type_counts = get_entity_type_counts(df['entities'])
    entity_counts = get_counts_of_all_entity(df['entities'], 25)
    persons = get_counts_of_spec_entity(df['entities'], 'PERSON')
    GPEs = get_counts_of_spec_entity(df['entities'], 'GPE')
    ORGs = get_counts_of_spec_entity(df['entities'], 'ORG')
    FACs = get_counts_of_spec_entity(df['entities'], 'FAC')
    LOCs = get_counts_of_spec_entity(df['entities'], 'LOC')
    location_counts = Counter(GPEs + ORGs + FACs + LOCs + persons)
    print(location_counts)
    visits = pd.DataFrame(london_locations).transpose()
    visits.reset_index(inplace=True)
    visits.rename(columns={0: 'latitude', 1: 'longitude', 'index': 'location_name'}, inplace=True)
    location_counts = pd.DataFrame.from_dict(location_counts, orient='index').reset_index()
    location_counts.rename(columns={0: 'count', 'index': 'location_name'}, inplace=True)

    visits = pd.merge(visits, location_counts)
    print(visits)
    name_merging = {
        'Parliament': 'Westminster',
        'Westminster Hall': 'Westminster',
        'the House of Lords': 'Westminster',
        'White hall': 'Whitehall',
        'Fleete Street': 'Fleet Street',
        'Lumbard Street': 'Lombard Street',
    }
    visits.replace(name_merging, inplace=True)
    visits = visits.groupby('location_name', axis=0, as_index=False).agg(
        {'latitude': 'mean', 'longitude': 'mean', 'count': 'sum', })
    return entity_type_counts, entity_counts, visits


if __name__ == "__main__":
    entries = pd.read_csv('entries.csv')
    entity_type_counts, entity_counts, visits = extract_entities(entries)
    print(visits)
    plot_locations(visits, range_color=[0, 50], zoom=9)



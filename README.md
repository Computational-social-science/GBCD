# GBCD

The global competition for human capital is fuelled by intricate brain circulation dynamics, where individuals with specialized skills traverse geographic, organizational, and national boundaries to address workforce demands. However, a comprehensive framework for integrating and interpreting heterogeneous data on global brain circulation remains elusive. Here we introduce the Global Brain Circulation Dynamics (GBCD) corpus, a longitudinally integrated repository of geo-information encompassing 223 countries/regions from 2000 to 2024. Garnered from diachronic narrative texts, the GBCD corpus provides granular insights into transnational brain circulation patterns and their interconnections with sociocultural progress. Continuously updated to reflect spatiotemporal dynamics, the GBCD corpus serves as a definitive reference for real-time and ex-post analysis of global brain circulation. Our analysis reveals two pivotal findings: (i) narrative brain circulation closely mirrors physical brain mobility, and (ii) geopolitical relations and spatiotemporal dynamics exhibit distinct patterns across countries/regions. The GBCD corpus establishes a novel benchmark for examining spatiotemporal brain circulation worldwide, empowering policymakers to develop evidence-based strategies for attracting and retaining human capital in rapidly evolving global landscape.

## Corpus
The GBCD corpus is a comprehensive dataset comprising 2,904,663,710 tokens, structured into two distinct corpora: diachronic and synchronic. The corpus encompasses 1,273,626 entries related to brain circulation features, with the diachronic corpus accounting for 1,132,674 entries that span a 24-year period (2000-2024). Notably, the diachronic corpus is continuously updated in real-time, ensuring the data remains current and relevant for both real-time and ex-post analyses of brain circulation. In contrast, the synchronic corpus contains 140,952 entries, deliberately excluding timestamp features to facilitate synchronic research.
<table>
  <tr>
    <th>Corpus</th>
    <th>Entry Count</th>
    <th>Processed Token Count</th>
    <th>Token Count</th>
    <th>Sentence Count</th>
  </tr>
  <tr>
    <td>Diachronic corpus</td>
    <td>1,132,674</td>
    <td>2,087,930,788</td>
    <td>422,954,074</td>   
    <td>36,914,973</td>   
  </tr>
  <tr>
    <td>Synchronic corpus</td>
    <td>140,952</td>
    <td>816,732,922</td>
    <td>158,891,392</td>
    <td>12,250,558</td>
  </tr>
</table>



## Data record
The corpus captures key attributes relevant to brain circulation, including origin, destination, diachronic narrative text, URL, and timestamp. Notably, geographic entities are mapped to the global country or region level, facilitating the analysis of transnational brain circulation.  Each country or region is accompanied by Countrycode, ISO2, and ISO3 identifiers, enabling multidimensional organization of brain circulation data. Furthermore, we distinguish between origin and destination in geographic entities related to circulation flow, allowing for the representation of brain gain and brain drain, and providing insights into bilateral brain circulation between countries.


### Summary information about the GBCD corpus
<table>
  <tr>
    <th>Data Label</th>
    <th>Data Description</th>
    <th>Data Type</th>
  </tr>
  <tr>
    <td>circulation id</td>
    <td>Unique circulation behaviour text identification</td>
    <td>int</td>
  </tr>
  <tr>
    <td>content</td>
    <td>The narrative text content in the web address</td>
    <td>long text</td>
  </tr>
  <tr>
    <td>countrycode</td>
    <td>ISO country code</td>
    <td>string</td>
  </tr>
  <tr>
    <td>countrycode</td>
    <td>ISO country code</td>
    <td>string</td>
  </tr>
  <tr>
    <td>URL</td>
    <td>Source links to transfer narrative text, usually pointing to web pages and domain names</td>
    <td>string</td>
  </tr>
  <tr>
    <td>timestamp</td>
    <td>Month and Year of transfer behaviour described in the text</td>
    <td>date object</td>
  </tr>
  <tr>
    <td>sampling</td>
    <td>The collection timestamp of the text data in the source dataset</td>
    <td>date object</td>
  </tr>
  <tr>
    <td>iso2code</td>
    <td>Country ISO 2 letter code</td>
    <td>string</td>
  </tr>
  <tr>
    <td>iso3code</td>
    <td>Country ISO 3 letter code</td>
    <td>string</td>
  </tr>
  <tr>
    <td>origin</td>
    <td>The origin of the circulation behaviour, expressed as geopolitical entity, including country or region</td>
    <td>string</td>
  </tr>
  <tr>
    <td>destination</td>
    <td>The destination of circulation behaviour, expressed as geopolitical entity, including country or region</td>
    <td>string</td>
  </tr>
</table>



### Geographic entity criteria
The GBCD corpus spans 223 countries and regions worldwide, encompassing 193 UN member states, one observer state, and 29 non-sovereign island territories.Our national geographic divisions adhere to methods endorsed by the United Nations Statistics Division for international statistical data collection, ensuring consistency and compatibility with global standards.
* **Member State of the United Nations:** refers to a sovereign country that has been officially admitted to the United Nations (UN) and holds full membership status. Member States enjoy voting rights, participate in all UN activities, and are bound by the principles outlined in the UN Charter.
  - [UN Membership](https://www.un.org/zh/about-us/member-states)
* **Non-Member Observer State of the United Nations:** refers to an entity recognized by the United Nations General Assembly that has observer status, granting it certain privileges and participation rights in UN activities, but without full membership or voting rights in the General Assembly.
  - [Palestine](https://documents.un.org/doc/undoc/gen/n12/479/73/pdf/n1247973.pdf)
* **Regions without Internationally Recognized Sovereignty:** refer to territories or regions that declare themselves as independent or autonomous but lack widespread recognition as sovereign states under international law or by the global community, including the United Nations.
  * Territories: Western Sahara,South Ossetia,Transnistria,Northern Cyprus,Kosovo,Western Sahara,Abkhazia,Catalonia,Transnistria,Nagorno-Karabakh,Artsakh Republic
  * Islands: Abah Island,Kuril Islands,Malvinas Islands,Falkland Islands,Hanish Islands,Palmyra Atoll,Spratly Islands,Barbuda Islands,Chafarinas Islands,Saint Helena Islands,Seto Islands,Aguinas Islands,Kirkira Islands,Gulf of Gdańsk Islands,Kuril Islands,Falkland Islands,Spratly Islands,Paracel Islands
<br/><br/>



## Data mining
### Domain name distribution by continents and fields.<br/>
![image](https://github.com/Computational-social-science/GBCD/blob/main/4.Domain%20name%20distribution/domain_name_distribution.svg)
<br/>
### Geographical heterogeneity of national brain circulation frequency.<br/>
![image](https://github.com/Computational-social-science/GBCD/blob/main/5.Geographical%20heterogeneity/geographical_heterogeneity.png)
<br/>
### Geographical trajectory network of transnational brain circulation.<br/>
<img src="6.Geographical%20trajectory%20network/output.png" style="zoom: 80%;" />
<br/>
### Dynamic indicators of national brain circulation flux.<br/>
![image](https://github.com/Computational-social-science/GBCD/blob/main/7.Dynamic%20indicators/dynamic_indicators.png)
<br/>
## Usage Notes
The GBCD corpus enables the comprehensive assessment and characterization of global brain circulation, facilitating planning and analysis at the national and geographic levels. To ensure high data quality and extensive geographic coverage, specific names, materials, and map layouts have been employed. It is essential to note that these choices do not imply any endorsement or stance by the authors or their respective countries regarding the legal status of any nation, territory, or region. Additionally, the depiction of borders and boundaries on the maps is purely indicative and does not signify formal recognition or acceptance by the publisher. The maps and database are intended to provide a neutral representation of geographic information, and any interpretation or inference of political boundaries or affiliations is explicitly excluded.


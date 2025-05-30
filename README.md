# Global Brain Circulation Dynamics (GBCD) corpus
The global competition for human capital is fuelled by intricate brain circulation dynamics, where individuals with specialized skills traverse geographic, organizational, and national boundaries to address workforce demands. However, a comprehensive framework for integrating and interpreting heterogeneous data on global brain circulation remains elusive. 

Here we introduce [the Global Brain Circulation Dynamics (GBCD) corpus](https://doi.org/10.6084/m9.figshare.28031471), a longitudinally integrated repository of geo-information encompassing 223 countries/regions from 2000 to 2024. <br/>

Garnered from diachronic narrative texts, the [GBCD corpus](https://doi.org/10.6084/m9.figshare.28031471) provides granular insights into transnational brain circulation patterns and their interconnections with sociocultural progress. Continuously updated to reflect spatiotemporal dynamics, the [GBCD corpus](https://doi.org/10.6084/m9.figshare.28031471) serves as a definitive reference for real-time and ex-post analysis of global brain circulation. Our analysis reveals two pivotal findings:
<br/> 
*   narrative brain circulation closely mirrors physical brain mobility
*   geopolitical relations and spatiotemporal dynamics exhibit distinct patterns across countries/regions<br/>

The [GBCD corpus](https://doi.org/10.6084/m9.figshare.28031471) establishes a novel benchmark for examining spatiotemporal brain circulation worldwide, empowering policymakers to develop evidence-based strategies for attracting and retaining human capital in rapidly evolving global landscape.<br/>

## Corpus
The [GBCD corpus](https://doi.org/10.6084/m9.figshare.28031471) is a comprehensive dataset comprising 2,904,663,710 tokens, structured into two distinct corpora: diachronic and synchronic. 

The corpus encompasses 1,764,234 entries related to brain circulation features, with the diachronic corpus accounting for 1,311,616 entries that span a 24-year period (2000-2024). Notably, the diachronic corpus is continuously updated in real-time, ensuring the data remains current and relevant for both real-time and ex-post analyses of brain circulation. In contrast, the synchronic corpus contains 452,618 entries, deliberately excluding timestamp features to facilitate synchronic research.

<table>
  <tr>
    <th>Version</th>
    <th>Update Time</th>
    <th>Corpus</th>
    <th>Entry Count</th>
    <th>Processed Token Count</th>
    <th>Token Count</th>
    <th>Sentence Count</th>
  </tr>
  <tr>
    <td rowspan="2">V1.0</td>
    <td rowspan="2">2024-8-29</td>
    <td>Diachronic corpus</td>
    <td>623,072</td>
    <td>1,134,253,949</td>
    <td>422,954,074</td>   
    <td>16,914,973</td>   
  </tr>
  <tr>
    <td>Synchronic corpus</td>
    <td>348,508</td>
    <td>606,015,828</td>
    <td>158,891,392</td>
    <td>11,250,558</td>
  </tr>

  <tr>
    <td rowspan="2">V2.0</td>
    <td rowspan="2">2024-12-16</td>
    <td>Diachronic corpus</td>
    <td>1,111,644</td>
    <td>2,087,930,788</td>
    <td>707,785,647</td>   
    <td>38,900,418</td>   

    
  </tr>
  <tr>
    <td>Synchronic corpus</td>
    <td>452,618</td>
    <td>816,732,922</td>
    <td>328,842,410</td>
    <td>19,253,646</td>
  </tr>
</table>


## Data record
The [GBCD corpus](https://doi.org/10.6084/m9.figshare.28031471) captures key attributes relevant to brain circulation, including origin, destination, diachronic narrative text, URL, and timestamp. Notably, geographic entities are mapped to the global country or region level, facilitating the analysis of transnational brain circulation.  Each country or region is accompanied by Countrycode, ISO2, and ISO3 identifiers, enabling multidimensional organization of brain circulation data. Furthermore, we distinguish between origin and destination in geographic entities related to circulation flow, allowing for the representation of brain gain and brain drain, and providing insights into bilateral brain circulation between countries/rigions.

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


#### Regions without internationally recognized sovereignty:
These regions do not possess formal recognition or authority under international law, meaning they lack official ISO codes and CountryCodes. As a result, they are not represented in global standards used for identifying sovereign states.

#### Geospatial representation:
To delineate the geographic boundaries of such regions, we rely on Polygon-type geospatial data. This approach allows for the precise definition of the spatial extent of these areas, even in the absence of formal sovereignty. The polygon format enables the mapping of complex territorial claims or disputed regions, capturing their exact geographic features.

#### Structured information for countries/regions:
Detailed and structured data related to these regions, as well as fully recognized countries, can be accessed in the [Supplementary information
](https://github.com/Computational-social-science/GBCD/blob/main/Supplementary%20information/Countries%26Regions%20Information.xlsx). This repository includes comprehensive information about their geographic, political, and other relevant attributes, offering an in-depth look at the regions' boundaries, history, and territorial disputes.

### Geographic entity criteria
The [GBCD corpus](https://doi.org/10.6084/m9.figshare.28031471) spans 223 countrie/regions worldwide, encompassing 193 UN member states, one observer state, and 29 non-sovereign island territories.Our national geographic divisions adhere to methods endorsed by the United Nations Statistics Division for international statistical data collection, ensuring consistency and compatibility with global standards.
* **Member State of the United Nations:** refers to a sovereign country that has been officially admitted to the United Nations (UN) and holds full membership status. Member States enjoy voting rights, participate in all UN activities, and are bound by the principles outlined in the UN Charter.
  - [UN Membership](https://www.un.org/en/about-us/member-states)
* **Non-Member Observer State of the United Nations:** refers to an entity recognized by the United Nations General Assembly that has observer status, granting it certain privileges and participation rights in UN activities, but without full membership or voting rights in the General Assembly.
  - [Palestine](https://documents.un.org/doc/undoc/gen/n12/479/74/pdf/n1247974.pdf)
* **Territories and Islands without Internationally Recognized Sovereignty:** refer to territories and islands that declare themselves as independent or autonomous but lack widespread recognition as sovereign states under international law or by the global community, including the United Nations.
  - [Territories](https://github.com/Computational-social-science/GBCD/blob/main/Supplementary%20information/Geographic%20entity%20criteria.xlsx)
  - [Islands](https://github.com/Computational-social-science/GBCD/blob/main/Supplementary%20information/Geographic%20entity%20criteria.xlsx)
<br/><br/>


## Data mining
Leveraging data mining techniques on the [GBCD corpus](https://doi.org/10.6084/m9.figshare.28031471) enables researchers to map and characterize the brain circulation patterns of skilled professionals across different countries. Further more, researchers can gain a deeper understanding of the complex dynamics underlying brain circulation and make informed decisions to address the challenges.  This study highlights the potential of data-driven approaches to inform policy and promote more effective brain circulation strategies.
### Domain name distribution by continents and fields.<br/>
![image](https://github.com/Computational-social-science/GBCD/blob/main/4.Domain%20name%20distribution/domain_name_distribution.svg)
<br/>
### Geographical heterogeneity of national brain circulation frequency.<br/>
![image](https://github.com/Computational-social-science/GBCD/blob/main/5.Geographical%20heterogeneity/geographical_heterogeneity.png)
<br/>
### Geographical trajectory network of transnational brain circulation.<br/>
![image](https://github.com/Computational-social-science/GBCD/blob/main/6.Geographical%20trajectory%20network/geographical_trajectory_network.png)
<br/>
### Dynamic indicators of national brain circulation flux.<br/>
![image](https://github.com/Computational-social-science/GBCD/blob/main/7.Dynamic%20indicators/dynamic_indicators.svg)
<br/>
## Usage Notes
The [GBCD corpus](https://doi.org/10.6084/m9.figshare.28031471) enables the comprehensive assessment and characterization of global brain circulation, facilitating planning and analysis at the national and geographic levels. To ensure high data quality and extensive geographic coverage, specific names, materials, and map layouts have been employed. It is essential to note that these choices do not imply any endorsement or stance by the authors or their respective countries regarding the legal status of any nation, territory, or region. 

Additionally, the depiction of borders and boundaries on the maps is purely indicative and does not signify formal recognition or acceptance by the publisher. The maps and database are intended to provide a neutral representation of geographic information, and any interpretation or inference of political boundaries or affiliations is explicitly excluded.
## Citing this work
Zhiwen Hu, Yang Qiu, Haihua Jiang, Xiao Ma, Lv Han, Saihua Lei & Haojia Niu. Unveiling the Spatiotemporal Dynamics of Global Brain Circulation: A Comprehensive Corpus (2000–2024). _Scientific Data_. DOI: 10.1038/s41597-025-05268-2

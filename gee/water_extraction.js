// Sentinel-2 water body extraction over the Leshan study area.
//
// The script builds a cloud-masked Sentinel-2 surface reflectance composite,
// appends the NDWI and MNDWI indices, and exports three rasters:
//   1. Leshan_AllBands      - 7-channel model input (B2, B3, B4, B8, B11, NDWI, MNDWI)
//   2. Leshan_RGB           - true colour preview
//   3. Leshan_JRC_WaterLabel - initial binary water extent from the JRC global surface
//      water dataset, later inspected and corrected in QGIS
//
// Run this script in the Google Earth Engine Code Editor and start the three
// export tasks from the Tasks panel.

// ---------------------------------------------------------------------------
// Study area: the confluence of the Min, Dadu and Qingyi rivers.
// ---------------------------------------------------------------------------
var studyArea = ee.Geometry.Rectangle([103.40, 29.20, 104.10, 29.80]);
Map.centerObject(studyArea, 11);
Map.addLayer(studyArea, {color: 'red'}, 'Study area');
print('Study area (km2):', studyArea.area().divide(1e6));

// ---------------------------------------------------------------------------
// Select low-cloud Sentinel-2 scenes from late summer 2025.
// Five scenes met the cloud-cover criterion in this study.
// ---------------------------------------------------------------------------
var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(studyArea)
  .filterDate('2025-08-01', '2025-09-30')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10));

print('Candidate scenes:', collection.size());

var best = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first();
print('Lowest cloud cover (%):', best.get('CLOUDY_PIXEL_PERCENTAGE'));

Map.addLayer(
  best.select(['B4', 'B3', 'B2']).clip(studyArea),
  {min: 0, max: 3000, gamma: 1.4},
  'Clearest scene (RGB)'
);

// ---------------------------------------------------------------------------
// Cloud and shadow masking using both QA60 and the scene classification layer.
// ---------------------------------------------------------------------------
function maskCloudsAndShadows(image) {
  var qa = image.select('QA60');
  var cloudBit = 1 << 10;
  var cirrusBit = 1 << 11;
  var qaMask = qa.bitwiseAnd(cloudBit).eq(0)
    .and(qa.bitwiseAnd(cirrusBit).eq(0));

  var scl = image.select('SCL');
  var sclMask = scl.neq(3)   // cloud shadow
    .and(scl.neq(8))         // medium probability cloud
    .and(scl.neq(9))         // high probability cloud
    .and(scl.neq(10));       // cirrus

  return image.updateMask(qaMask.and(sclMask));
}

var bands = ['B2', 'B3', 'B4', 'B8', 'B11'];

// Mask every candidate scene and build a mosaic that prefers the clearest
// observations, filling gaps left by clouds in a single scene.
var processed = collection.map(function (image) {
  return maskCloudsAndShadows(image)
    .select(bands)
    .divide(10000)
    .copyProperties(image, ['CLOUDY_PIXEL_PERCENTAGE']);
});

var mosaic = processed
  .sort('CLOUDY_PIXEL_PERCENTAGE')
  .mosaic()
  .clip(studyArea);

Map.addLayer(
  mosaic.select(['B4', 'B3', 'B2']),
  {min: 0, max: 0.3, gamma: 1.3},
  'Mosaic (RGB)'
);

// ---------------------------------------------------------------------------
// Water indices.
// ---------------------------------------------------------------------------
var ndwi = mosaic.normalizedDifference(['B3', 'B8']).rename('NDWI');

var mndwi = mosaic.expression(
  '(B3 - B11) / (B3 + B11)',
  {'B3': mosaic.select('B3'), 'B11': mosaic.select('B11')}
).rename('MNDWI');

var allBands = mosaic.addBands(ndwi).addBands(mndwi);
allBands = allBands.reproject({crs: 'EPSG:32648', scale: 10});
print('Channels: B2, B3, B4, B8, B11, NDWI, MNDWI');

Map.addLayer(
  ndwi,
  {min: -0.5, max: 0.5, palette: ['brown', 'white', 'blue']},
  'NDWI'
);
Map.addLayer(
  mndwi,
  {min: -0.5, max: 0.5, palette: ['brown', 'white', 'blue']},
  'MNDWI'
);

// ---------------------------------------------------------------------------
// Exports.
// ---------------------------------------------------------------------------
Export.image.toDrive({
  image: allBands.toFloat(),
  description: 'Leshan_AllBands',
  scale: 10,
  region: studyArea,
  crs: 'EPSG:32648',
  maxPixels: 1e9,
  folder: 'GEE_Water_Extraction'
});

Export.image.toDrive({
  image: mosaic.select(['B4', 'B3', 'B2']).toFloat(),
  description: 'Leshan_RGB',
  scale: 10,
  region: studyArea,
  crs: 'EPSG:32648',
  maxPixels: 1e9,
  folder: 'GEE_Water_Extraction'
});

var jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');
var jrcWater = jrc.select('max_extent')
  .clip(studyArea)
  .reproject({crs: 'EPSG:32648', scale: 10});

Export.image.toDrive({
  image: jrcWater.toFloat(),
  description: 'Leshan_JRC_WaterLabel',
  scale: 10,
  region: studyArea,
  crs: 'EPSG:32648',
  maxPixels: 1e9,
  folder: 'GEE_Water_Extraction'
});

load("CrimeDataKennedy.RData")

#plot(CrimesPP)
table(CrimesPP$marks$crime)
commercial_ppp <- subset(CrimesPP, crime == "commercial")
marks(commercial_ppp) <- NULL

commercial_ppp$n

Q <- quadscheme(commercial_ppp)
wg <- Q$w
label <- is.data(Q)
library(raster)

coordinates <- data.frame(x = Q$data$x, y = Q$data$y)
dummycoor <- data.frame(x = Q$dummy$x, y = Q$dummy$y)
extract_from_im <- function(im_obj, pts) {
  ras <- raster(as.im(im_obj))
  raster::extract(ras, pts)
}

transport_data      <- extract_from_im(SpatialCovariates$transport,      coordinates)
injuries_data       <- extract_from_im(SpatialCovariates$injuries,       coordinates)
construction_data   <- extract_from_im(SpatialCovariates$construction,   coordinates)
water_data          <- extract_from_im(SpatialCovariates$water,          coordinates)
pharmacies_data     <- extract_from_im(SpatialCovariates$pharmacies,     coordinates)
schools_data        <- extract_from_im(SpatialCovariates$schools,        coordinates)
health_data         <- extract_from_im(SpatialCovariates$health,         coordinates)
parks_data          <- extract_from_im(SpatialCovariates$parks,          coordinates)
markets_data        <- extract_from_im(SpatialCovariates$markets,        coordinates)
libraries_data      <- extract_from_im(SpatialCovariates$libraries,      coordinates)

transport_dummy     <- extract_from_im(SpatialCovariates$transport,      dummycoor)
injuries_dummy      <- extract_from_im(SpatialCovariates$injuries,       dummycoor)
construction_dummy  <- extract_from_im(SpatialCovariates$construction,   dummycoor)
water_dummy         <- extract_from_im(SpatialCovariates$water,          dummycoor)
pharmacies_dummy    <- extract_from_im(SpatialCovariates$pharmacies,     dummycoor)
schools_dummy       <- extract_from_im(SpatialCovariates$schools,        dummycoor)
health_dummy        <- extract_from_im(SpatialCovariates$health,         dummycoor)
parks_dummy         <- extract_from_im(SpatialCovariates$parks,          dummycoor)
markets_dummy       <- extract_from_im(SpatialCovariates$markets,        dummycoor)
libraries_dummy     <- extract_from_im(SpatialCovariates$libraries,      dummycoor)


truedata <- data.frame(
  x = coordinates$x,
  y = coordinates$y,
  transport = transport_data,
  injuries = injuries_data,
  construction = construction_data,
  water = water_data,
  pharmacies = pharmacies_data,
  schools = schools_data,
  health = health_data,
  parks = parks_data,
  markets = markets_data,
  libraries = libraries_data,
  label = 1,
  vol = wg[label],   # seperti struktur kamu sebelumnya
  int = 1
)

dummydata <- data.frame(
  x = dummycoor$x,
  y = dummycoor$y,
  transport = transport_dummy,
  injuries = injuries_dummy,
  construction = construction_dummy,
  water = water_dummy,
  pharmacies = pharmacies_dummy,
  schools = schools_dummy,
  health = health_dummy,
  parks = parks_dummy,
  markets = markets_dummy,
  libraries = libraries_dummy,
  label = -1,
  vol = wg[!label], 
  int = 0
)

df <- rbind(truedata, dummydata)
df <- na.omit(df)
write.csv(df,"crimenonscale.csv")


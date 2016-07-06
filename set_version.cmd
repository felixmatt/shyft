@echo off

git rev-list --count HEAD > REVISION

set /p minor=<REVISION

echo 3.0.%minor% > VERSION

echo ##teamcity[buildNumber '3.0.%minor%']

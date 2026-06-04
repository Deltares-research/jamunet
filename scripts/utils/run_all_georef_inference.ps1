param(
    [string]$PythonExe = 'python',
    [string]$OutputRoot = 'outputs/all_regions_all_years_georef',
    [string]$RegionFilter = '',
    [int]$TargetYear = 0
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $RepoRoot

$collection = 'JRC_GSW1_4_MonthlyHistory'
$catalogCandidates = @(
    (Join-Path $RepoRoot 'data\satellite\regions\region_catalog_model_ready.json'),
    (Join-Path $RepoRoot 'data\satellite\regions\region_catalog_raw_available.json'),
    (Join-Path $RepoRoot 'data\satellite\regions\region_catalog.json')
)
$catalogPath = $null
foreach ($candidate in $catalogCandidates) {
    if (Test-Path $candidate) {
        $catalogPath = $candidate
        break
    }
}
$datasetRoot = Join-Path $RepoRoot 'data\satellite\dataset_month3'
$logDir = Join-Path $RepoRoot (Join-Path $OutputRoot '_logs')
$summaryPath = Join-Path $RepoRoot (Join-Path $OutputRoot 'batch_summary.json')
$failuresPath = Join-Path $RepoRoot (Join-Path $OutputRoot 'failures.json')
$runExamplePath = Join-Path $RepoRoot 'scripts\inference\run_example.py'

if ($null -eq $catalogPath) {
    throw "Catalog not found. Checked: $($catalogCandidates -join ', ')"
}
if (-not (Test-Path $runExamplePath)) {
    throw "Inference entrypoint not found: $runExamplePath"
}

New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $OutputRoot) | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$catalog = Get-Content $catalogPath -Raw | ConvertFrom-Json
$regions = New-Object System.Collections.Generic.List[string]
foreach ($item in $catalog) {
    if ($null -ne $item.region_id) {
        $regions.Add([string]$item.region_id)
    }
    elseif ($null -ne $item.value) {
        foreach ($sub in $item.value) {
            if ($null -ne $sub.region_id) {
                $regions.Add([string]$sub.region_id)
            }
        }
    }
}
$regions = $regions | Sort-Object -Unique
if ($RegionFilter -ne '') {
    $requestedRegions = $RegionFilter.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
    $regions = $regions | Where-Object { $requestedRegions -contains $_ }
}

$attempted = 0
$successes = 0
$failures = New-Object System.Collections.Generic.List[object]
$startedAt = Get-Date

foreach ($region in $regions) {
    $regionDir = Join-Path $datasetRoot ("$collection`_$region")
    if (-not (Test-Path $regionDir)) {
        $failures.Add([pscustomobject]@{
            region_id = $region
            target_year = $null
            reason = 'Missing dataset_month3 folder'
        })
        continue
    }

    $years = Get-ChildItem $regionDir -Filter '*.tif' |
        ForEach-Object { [int]($_.BaseName.Split('_')[0]) } |
        Sort-Object -Unique

    if ($years.Count -le 4) {
        $failures.Add([pscustomobject]@{
            region_id = $region
            target_year = $null
            reason = 'Not enough years to build 4->1 samples'
        })
        continue
    }

    $targetYears = $years[4..($years.Count - 1)]
    if ($TargetYear -gt 0) {
        $targetYears = $targetYears | Where-Object { $_ -eq $TargetYear }
    }

    if ($targetYears.Count -eq 0) {
        continue
    }

    foreach ($year in $targetYears) {
        $regionOutputDir = Join-Path $RepoRoot (Join-Path $OutputRoot $region)
        $expectedProbGeoref = Join-Path $regionOutputDir ("prediction_probabilities_georef_$region`_$year.tif")
        $expectedVisGeoref = Join-Path $regionOutputDir ("prediction_binary_vis_georef_$region`_$year.tif")

        if ((Test-Path $expectedProbGeoref) -and (Test-Path $expectedVisGeoref)) {
            $attempted += 1
            $successes += 1
            Write-Host ("[$attempted] Skipping completed region=$region year=$year")

            $summary = [pscustomobject]@{
                started_at = $startedAt.ToString('s')
                updated_at = (Get-Date).ToString('s')
                total_regions = $regions.Count
                attempted_runs = $attempted
                successful_runs = $successes
                failed_runs = $failures.Count
                output_root = $OutputRoot
            }
            $summary | ConvertTo-Json | Set-Content -Path $summaryPath
            $failures | ConvertTo-Json -Depth 5 | Set-Content -Path $failuresPath
            continue
        }

        $attempted += 1
        $logPath = Join-Path $logDir ("$region`_$year.log")
        Write-Host ("[$attempted] Running region=$region year=$year")

        try {
            $commandLine = '"' + $PythonExe + '" "' + $runExamplePath + '" --region ' + $region + ' --target-year ' + $year + ' --output-dir ' + $OutputRoot + ' 2>&1'
            cmd.exe /d /c $commandLine |
                Tee-Object -FilePath $logPath | Out-Host
            if ($LASTEXITCODE -eq 0) {
                $successes += 1
            }
            else {
                $failures.Add([pscustomobject]@{
                    region_id = $region
                    target_year = $year
                    reason = "Exit code $LASTEXITCODE"
                })
            }
        }
        catch {
            $_ | Out-String | Tee-Object -FilePath $logPath -Append | Out-Host
            $failures.Add([pscustomobject]@{
                region_id = $region
                target_year = $year
                reason = $_.Exception.Message
            })
        }

        $summary = [pscustomobject]@{
            started_at = $startedAt.ToString('s')
            updated_at = (Get-Date).ToString('s')
            total_regions = $regions.Count
            attempted_runs = $attempted
            successful_runs = $successes
            failed_runs = $failures.Count
            output_root = $OutputRoot
        }
        $summary | ConvertTo-Json | Set-Content -Path $summaryPath
        $failures | ConvertTo-Json -Depth 5 | Set-Content -Path $failuresPath
    }
}

$sampleGeoref = Get-ChildItem (Join-Path $RepoRoot $OutputRoot) -Recurse -Filter '*_georef_*.tif' |
    Select-Object -First 10 -ExpandProperty FullName

$finalSummary = [pscustomobject]@{
    started_at = $startedAt.ToString('s')
    finished_at = (Get-Date).ToString('s')
    total_regions = $regions.Count
    attempted_runs = $attempted
    successful_runs = $successes
    failed_runs = $failures.Count
    output_root = $OutputRoot
    sample_georef_files = $sampleGeoref
}

$finalSummary | ConvertTo-Json -Depth 5 | Set-Content -Path $summaryPath
$failures | ConvertTo-Json -Depth 5 | Set-Content -Path $failuresPath
$finalSummary | ConvertTo-Json -Depth 5 | Write-Output

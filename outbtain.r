# Kaggle competition outbrain-click-prediction

# NOTE: this script expects that libffm be built and executables be placed in the working directory.
# NOTE: I had to modify the xgb source code to ignore the 10M rows limitation on exact modeling.

needs(data.table, xgboost, Metrics, Matrix, glmnet, caret, Hmisc, lightgbm)

debug.mode = F
valid.mode = F

do.load     = T
do.baseline = F
do.ffm      = T
do.xgb      = T
do.lgb      = F
do.submit   = T

ffm.run.select = 'cv' # { final, valid, cv }
xgb.run.select = 'final' # { final, valid, stack }

xgbd.from.disk = F

rng.seed = 123
submission.id = 4

#tmpdir = 'C:/TEMP/kaggle/outbrain-click-prediction'
tmpdir = 'tmp'

if (debug.mode) {
  ffm.train.fn     = paste0(tmpdir, '/train-sample.ffm'         )
  ffm.valid.fn     = paste0(tmpdir, '/valid-sample.ffm'         )
  ffm.test.fn      = paste0(tmpdir, '/test-sample.ffm'          )
  ffm.model.fn     = paste0(tmpdir, '/ffm-sample.model'         )
  ffm.preds.fn     = paste0(tmpdir, '/ffm-sample.preds'         )
  ffm.stackedv.fn  = paste0(tmpdir, '/ffm-stackedv-sample.RData')
  ffm.stacked.fn   = paste0(tmpdir, '/ffm-stacked-sample.RData' )
} else {
  ffm.train.fn     = paste0(tmpdir, '/train.ffm'         )
  ffm.valid.fn     = paste0(tmpdir, '/valid.ffm'         )
  ffm.test.fn      = paste0(tmpdir, '/test.ffm'          )
  ffm.model.fn     = paste0(tmpdir, '/ffm.model'         )
  ffm.preds.fn     = paste0(tmpdir, '/ffm.preds'         )
  ffm.stackedv.fn  = paste0(tmpdir, '/ffm-stackedv.RData')
  ffm.stacked.fn   = paste0(tmpdir, '/ffm-stacked.RData' )
}

validate = function(dat, labels, valid.mask, preds) {
  vld = data.table(display_id = dat[valid.mask, display_id], label = labels[valid.mask], pred = preds)
  setorder(vld, display_id, -pred)
  vld[, w := 1 / (1:.N), by = display_id]
  vld[w < 1/12, w := 0]
  map12 = vld[label == 1, mean(w)]
  cat('Validation map12 =', map12, '\n')
}

generate.submission = function(submission.id, preds) {
  sbmt = fread('input/clicks_test.csv')
  setorder(sbmt, display_id, ad_id) # NOTE: this is how we ordered "dat", and so this is the order in which "preds" are given
  sbmt[, pred := preds]
  setorder(sbmt, display_id, -pred)
  sbmt[, rnk := 1:.N, by = display_id]
  sbmt = sbmt[rnk <= 12] # there are no displays with more than 12 ads, but ok
  submission = sbmt[, .(ad_id = paste(ad_id, collapse = ' ')), by = display_id]
  fwrite(submission, paste0('sbmt-', submission.id, '.csv'))
  zip(paste0('sbmt-', submission.id, '.zip'), paste0('sbmt-', submission.id, '.csv'))
}

set.seed(rng.seed)

if (do.load) {
  cat(date(), 'Loading data\n')
  
  if (debug.mode) {
    cat(date(), 'NOTE: Working on sampled data for quick dev/dbg\n')
    load(paste0(tmpdir, '/ppdata2-sample.RData')) # => dat
  } else {
    load(paste0(tmpdir, '/ppdata2.RData')) # => dat
  }
  
  setDT(dat) # it turns out that when loaded from disk, data.table needs to be manually "helped" for some of its behavior to work properly!
  test.mask = is.na(dat$clicked)
  
  if (valid.mode) {
    # Following the split suggested on the forum (haven't thought about it too much, maybe worth doing so)
    dat.days = yday(dat$time.utc - 4 * 3600) - 165
    udids = unique(dat[!test.mask & dat.days <= 10, display_id])
    valid.mask = !test.mask & ((dat.days > 10) | (dat$display_id %in% sample(udids, round(0.2 * length(udids)))))
    train.mask = !test.mask & !valid.mask
    
    cat(date(), 'Sample sizes: train', sum(train.mask), 'valid', sum(valid.mask), 'test', sum(test.mask), '\n')
    
    rm(dat.days, udids); #gc()
  } else {
    train.mask = !test.mask
    valid.mask = NULL
    cat(date(), 'Sample sizes: train', sum(train.mask), 'test', sum(test.mask), '\n')
  }
  
  labels = dat$clicked
  dat[, clicked := NULL]
}

if (do.baseline) {
  # Our baseline is the marginal ad score
  cat(date(), 'Baseline model\n')
  baseline = dat[train.mask, .(ad_id, label = labels[train.mask])]
  baseline = baseline[, .(adcount = .N, adscore = sum(label) / (6 + .N)), by = ad_id]
  
  if (valid.mode) {
    preds = merge(dat[valid.mask, .(ad_id)], baseline, by = 'ad_id', all.x = T, sort = F)$adscore
    preds[is.na(preds)] = mean(baseline[adcount == 1, adscore]) # FIXME is this optimal?
    validate(dat, labels, valid.mask, preds)
  }
  
  rm(baseline); gc()
}

if (do.ffm) {
  # According to https://www.andrew.cmu.edu/user/yongzhua/conferences/ffm.pdf
  # The meaning of a "field" for this kind of problem is the original categorical feature that the
  # final (say OHE) feature comes from, and the index is the coded level, the value in this case would
  # be 0 or 1 (hot/not). FFM learns a regularized version of the second-order logistic regression,
  # in which 
  # 1. Only interactions of features from different fields are modeled (naturally, because within a
  #    field the features are mutually exclusive).
  # 2. Let f1 and f2 be fields, and let j1 and j2 be indexes in f1 and f2, respectively. Then for 
  #    every combination of f1, f2, and j1, the set of coefficients of pairs (x_f1_j1, x_f2_j2) are
  #    constrained to be equal for all j2. This is kind of reasonable, and greatly simplifies the
  #    situation both statistically and computaitonally.
  # 3. We add L1 and L2 weight vector penalty terms.
  
  #
  # FFM parameters:
  #
  
  ffm.k = 8      # can try higher values (uses more memory. Contrary to the comments on the forum, I did not see an improvement with higher value on the sampled data. Perhaps that is different on the full data?)
  ffm.l = 0.0001 # can probably tune (regularization)
  ffm.r = 0.2    # can probably tune (step size)
  ffm.s = 16     # number of threads
  
  ffm.t.max   = 100 # NOTE: we will early stop
  ffm.t.final = ifelse(debug.mode, 7, 15) # FIXME make sure this matches the validation result above. Should it be higher for the full trainset?
  
  # other things I can try: --no-norm, --no-rand, --on-disk if out of RAM
  
  # FIXME: a better implementation of this that would be less memory intensive is to first figure 
  # out the fields and indexes, save this aside, and then transform and write to disk in row batches.
  #
  # In the meantime, I can save memory by canibalizing dat (inline by column transforms to create dffm)
  
  ds.to.libffm = function(dat, labels) {
    # The input file format for libFFM is
    # label field1:index1:value1 field2:index2:value2 ...
    # where label is either 0 or 1, and the remaining is a variable length list of nonzero feature
    # identifiers and their values. "Field"s are indexes (starting from 0) of feature groups (for OHE:
    # groups identify the original feature that was OHE), "index"es are feature identifiers within
    # groups (for OHE: category number in OHE order, starting from 0), and "value"s are... values
    # (for OHE: always 1).
    
    fld.offset = -1
    idx.offset = 0
    ttl.idx = 0
    
    ffm.encode.cat = function(x, newfld) {
      if (newfld) {
        fld.offset <<- fld.offset + 1
        idx.offset <<- 0
      }
      isnax = is.na(x)
      index = as.integer(x) - 1
      index[isnax] = nlevels(x)
      encx = paste(fld.offset, idx.offset + index, 1, sep = ':')
      idx.delta = nlevels(x) + any(isnax)
      idx.offset <<- idx.offset + idx.delta
      ttl.idx <<- ttl.idx + idx.delta
      return (encx)
    }
    
    ffm.encode.quaquaqua = function(x, newfld) {
      if (newfld) {
        fld.offset <<- fld.offset + 1
        idx.offset <<- 0
      }
      isnax = is.na(x)
      index = ifelse(isnax, idx.offset + 1, idx.offset)
      value = ifelse(isnax, 1, round(x, digits = 4)) # FIXME does digits matter here? how many are enough, how many are supported?
      encx = paste(fld.offset, index, value, sep = ':')
      idx.delta = 1 + any(isnax)
      idx.offset <<- idx.offset + idx.delta
      ttl.idx <<- ttl.idx + idx.delta
      return (encx)
    }
    
    ffm.encode.quaqua = function(x, newfld) {
      # Bin to some manageable number of categories
      m = uniqueN(x)
      mb = ifelse(m < 120, m, min(120, floor(m / 6)))
      ffm.encode.cat(cut2(x, g = mb), newfld)
    }
    
    ffm.encode.qua = function(x, newfld) { 
      # acutally treat as categorical (this works if the quantiative feature only takes a handful of unique values)
      ffm.encode.cat(as.factor(x), newfld)
    }
    
    labels[is.na(labels)] = 0
    dffm = data.table(label = labels)
    
    # FIXME do I need to manually make this thing full rank by encoding contrasts (eliminating levels etc)?
    # ?? It seems absurd that this model has no main effects. It only models main effects indirectly
    # via interactions between correlated features ("fields").
    
    if (1) {
      cat(date(), 'Main quantitatives\n')
      dffm[, leak              := ffm.encode.qua(dat$leak             , T)]
      dffm[, uuid_campaign     := ffm.encode.qua(dat$uuid_campaign    , T)]
      dffm[, uuid_domain       := ffm.encode.qua(dat$uuid_domain      , T)]
      dffm[, uuid_subdomain    := ffm.encode.qua(dat$uuid_subdomain   , T)]
      dffm[, uuid_cat          := ffm.encode.qua(dat$uuid_cat         , T)]
      dffm[, uuid_ent          := ffm.encode.qua(dat$uuid_ent         , T)]
      #dffm[, uuid_top          := ffm.encode.qua(dat$uuid_tpo         , T)]
      dffm[, disp_ad_cat_match := ffm.encode.qua(dat$disp_ad_cat_match, T)]
      dffm[, disp_ad_ent_match := ffm.encode.qua(dat$disp_ad_ent_match, T)]
      dffm[, disp_ad_top_match := ffm.encode.qua(dat$disp_ad_top_match, T)]
      #dffm[, is.office.hour    := ffm.encode.qua(dat$is.office.hour   , F)] # no good??
      #gc()
    }
    
    if (0) {
      cat(date(), 'More quantitatives\n')
      
     #dffm[, disp_ad_pub_atd      := ffm.encode.qua(dat$disp_ad_pub_atd     , F)] # no good?
     #dffm[, disp_ad_pub_std      := ffm.encode.qua(dat$disp_ad_pub_std     , F)]
     #dffm[, event_ad_pub_atd     := ffm.encode.qua(dat$event_ad_pub_atd    , F)] # no good?
     #dffm[, event_ad_pub_std     := ffm.encode.qua(dat$event_ad_pub_std    , F)] # unclear
     #dffm[, ad_target_popularity := ffm.encode.qua(dat$ad_target_popularity, F)] # no good?
      
     #dffm[, ad.display.idx       := ffm.encode.qua(dat$ad.display.idx      , F)]
      dffm[, display.size         := ffm.encode.qua(dat$display.size        , T)]
      
      dffm[, time := dat$time.utc - 4 * 3600]
     #dffm[, day   := ffm.encode.qua(yday(time) - 165       , T)] # this won't extrapolate
      dffm[, wday  := ffm.encode.qua(wday(time)             , T)] # this won't extrapolate
     #dffm[, hour1 := ffm.encode.qua(hour(time)             , T)]
      dffm[, hour2 := ffm.encode.qua((hour(time) - 12) %% 24, T)]
      dffm[, time := NULL]
    }
    
    if (0) { # These seem to make things much worse (no matter how I code them, no matter 
      # which of them I include). Why is that??
      cat(date(), 'Counts\n')
      
      dffm[, ad.evcnt                  := ffm.encode.qua(dat$ad.evcnt                 , T)]
      dffm[, ad.target.evcnt           := ffm.encode.qua(dat$ad.target.evcnt          , T)]
      dffm[, campaign_id.evcnt         := ffm.encode.qua(dat$campaign_id.evcnt        , T)]
      dffm[, advertiser_id.evcnt       := ffm.encode.qua(dat$advertiser_id.evcnt      , T)]
      dffm[, uuid.evcnt                := ffm.encode.qua(dat$uuid.evcnt               , T)]
      dffm[, country.evcnt             := ffm.encode.qua(dat$country.evcnt            , T)]
      dffm[, state.evcnt               := ffm.encode.qua(dat$state.evcnt              , T)]
      dffm[, geo_location.evcnt        := ffm.encode.qua(dat$geo_location.evcnt       , T)]
      dffm[, disp_docid.evcnt          := ffm.encode.qua(dat$disp_docid.evcnt         , T)]
      dffm[, uuid.ad.evcnt             := ffm.encode.qua(dat$uuid.ad.evcnt            , T)]
      dffm[, uuid.targ.evcnt           := ffm.encode.qua(dat$uuid.targ.evcnt          , T)]
      dffm[, uuid.camp.evcnt           := ffm.encode.qua(dat$uuid.camp.evcnt          , T)]
      dffm[, uuid.adv.evcnt            := ffm.encode.qua(dat$uuid.adv.evcnt           , T)]
      dffm[, country.ad.evcnt          := ffm.encode.qua(dat$country.ad.evcnt         , T)]
      dffm[, country.targ.evcnt        := ffm.encode.qua(dat$country.targ.evcnt       , T)]
      dffm[, country.camp.evcnt        := ffm.encode.qua(dat$country.camp.evcnt       , T)]
      dffm[, country.adv.evcnt         := ffm.encode.qua(dat$country.adv.evcnt        , T)]
      dffm[, state.ad.evcnt            := ffm.encode.qua(dat$state.ad.evcnt           , T)]
      dffm[, state.targ.evcnt          := ffm.encode.qua(dat$state.targ.evcnt         , T)]
      dffm[, state.camp.evcnt          := ffm.encode.qua(dat$state.camp.evcnt         , T)]
      dffm[, state.adv.evcnt           := ffm.encode.qua(dat$state.adv.evcnt          , T)]
      dffm[, disp_doc_domain.evcnt     := ffm.encode.qua(dat$disp_doc_domain.evcnt    , T)]
      dffm[, disp_doc_subdomain.evcnt  := ffm.encode.qua(dat$disp_doc_subdomain.evcnt , T)]
      dffm[, disp_doc_cat.evcnt        := ffm.encode.qua(dat$disp_doc_cat.evcnt       , T)]
      dffm[, disp_doc_ent.evcnt        := ffm.encode.qua(dat$disp_doc_ent.evcnt       , T)]
      dffm[, disp_doc_top.evcnt        := ffm.encode.qua(dat$disp_doc_top.evcnt       , T)]
      dffm[, ad_target_domain.evcnt    := ffm.encode.qua(dat$ad_target_domain.evcnt   , T)]
      dffm[, ad_target_subdomain.evcnt := ffm.encode.qua(dat$ad_target_subdomain.evcnt, T)]
      dffm[, ad_target_cat.evcnt       := ffm.encode.qua(dat$ad_target_cat.evcnt      , T)]
      dffm[, ad_target_ent.evcnt       := ffm.encode.qua(dat$ad_target_ent.evcnt      , T)]
      dffm[, ad_target_top.evcnt       := ffm.encode.qua(dat$ad_target_top.evcnt      , T)]
    }
    
    if (0) { # These seem to make things much worse. Why is that?
      cat(date(), 'Metafeatures\n')
      
      dffm[, ad.meta                   := ffm.encode.quaqua(dat$ad.meta                  , T)]
      dffm[, ad.target.meta            := ffm.encode.quaqua(dat$ad.target.meta           , T)]
      dffm[, campaign_id.meta          := ffm.encode.quaqua(dat$campaign_id.meta         , T)]
      dffm[, advertiser_id.meta        := ffm.encode.quaqua(dat$advertiser_id.meta       , T)]
      dffm[, ad.targetXplatform.meta   := ffm.encode.quaqua(dat$ad.targetXplatform.meta  , T)]
      dffm[, ad.targetXcountry.meta    := ffm.encode.quaqua(dat$ad.targetXcountry.meta   , T)]
      
      if (0) {
        # Add stacked deep trees
        if (debug.mode) {
          load(paste0(tmpdir, '/xgb-meta-sample.RData')) # => xgb.meta
        } else {
          load(paste0(tmpdir, '/xgb-meta.RData')) # => xgb.meta
        }
        
        dffm[, xgb.meta := ffm.encode.quaqua(xgb.meta, T)] # should I treat this as categorical?
      }
      
      gc()
    }
    
    if (1) {
      cat(date(), 'Main categoricals\n')
      
      dffm[, advertiser_id       := ffm.encode.cat(dat$advertiser_id      , T)]
      dffm[, campaign_id         := ffm.encode.cat(dat$campaign_id        , T)]
      dffm[, ad_target           := ffm.encode.cat(dat$ad_target          , T)]
      dffm[, ad_id               := ffm.encode.cat(dat$ad_id2             , T)]
      dffm[, ad_target_domain    := ffm.encode.cat(dat$ad_target_domain   , T)]
      dffm[, ad_target_subdomain := ffm.encode.cat(dat$ad_target_subdomain, T)]
      dffm[, ad_target_cat       := ffm.encode.cat(dat$ad_target_cat      , T)]
      dffm[, ad_target_ent       := ffm.encode.cat(dat$ad_target_ent      , T)]
      dffm[, ad_target_top       := ffm.encode.cat(dat$ad_target_top      , T)]
      
     #dffm[, uuid                := ffm.encode.cat(dat$uuid               , T)] # FIXME in the full data there are about 15M uuid. So this is only possible if I merge some classes somehow (e.g., at random)
      dffm[, country             := ffm.encode.cat(dat$country            , T)]
      dffm[, state               := ffm.encode.cat(dat$state              , T)]
      dffm[, geo_location        := ffm.encode.cat(dat$geo_location       , T)]
      dffm[, platform            := ffm.encode.cat(dat$platform           , T)]
      
      dffm[, disp_doc_domain     := ffm.encode.cat(dat$disp_doc_domain    , T)]
      dffm[, disp_doc_subdomain  := ffm.encode.cat(dat$disp_doc_subdomain , T)]
      dffm[, disp_doc_cat        := ffm.encode.cat(dat$disp_doc_cat       , T)]
      dffm[, disp_doc_ent        := ffm.encode.cat(dat$disp_doc_ent       , T)]
      dffm[, disp_doc_top        := ffm.encode.cat(dat$disp_doc_top       , T)]
      
      #gc()
    }
    
    if (0) {
      cat(date(), 'Ranks\n')
      
      # This doesn't seem to help? It's clearly important, but maybe as a main effect rather than
      # an interaction (but FFM doesn't have main effects!?)
      #dffm[, display.size := dat$display.size]
      
      d = copy(dat[, .(idx = 1:.N, display_id, ad_target_popularity, ad.evcnt, uuid.ad.evcnt, uuid.camp.evcnt)])
      
      setkey(d, display_id, ad_target_popularity)
      d[, ad_target_popularity.r := (1:.N) / .N, by = display_id]
      d[, ad_target_popularity := (ad_target_popularity > 0) * ad_target_popularity.r]
      d[, ad_target_popularity.r := NULL]
      setkey(d, idx)
      dffm[, ad_target_popularity.r := ffm.encode.qua(d$ad_target_popularity)]
      
      setkey(d, display_id, ad.evcnt)
      d[, ad.evcnt := (1:.N) / .N, by = display_id]
      setkey(d, idx)
      dffm[, ad.evcnt.r := ffm.encode.qua(d$ad.evcnt)]
      
      setkey(d, display_id, uuid.ad.evcnt)
      d[, uuid.ad.evcnt := as.numeric(uuid.ad.evcnt)]
      d[, uuid.ad.evcnt := (1:.N) / .N, by = display_id]
      setkey(d, idx)
      dffm[, uuid.ad.evcnt.r := ffm.encode.qua(d$uuid.ad.evcnt)]
      
      setkey(d, display_id, uuid.camp.evcnt)
      d[, uuid.camp.evcnt := as.numeric(uuid.camp.evcnt)]
      d[, uuid.camp.evcnt := (1:.N) / .N, by = display_id]
      setkey(d, idx)
      dffm[, uuid.camp.evcnt.r := ffm.encode.qua(d$uuid.camp.evcnt)]
      
      gc()
    }
    
    cat(date(), 'Total fields:', fld.offset, ', total indexes:', ttl.idx, '\n')
    
    return (dffm)
  }
  
  write.libffm = function(dffm, train.mask, valid.mask, test.mask, ffm.train.fn, ffm.valid.fn, ffm.test.fn) {
    if (0) {
      # EXPERIMENT: make this a case/control problem keeping only one (random) negative per display
      dffm.train = dffm[train.mask][order(dat$display_id[train.mask], -label, runif(sum(train.mask)))]
      idx = which(dffm.train$label == 1)
      fwrite(dffm.train[sample(c(idx, idx + 1))], ffm.train.fn, sep = ' ', col.names = F)
    } else {
      fwrite(dffm[train.mask], ffm.train.fn, sep = ' ', col.names = F)
    }
    
    if (!is.null(valid.mask)) {
      fwrite(dffm[valid.mask], ffm.valid.fn, sep = ' ', col.names = F)
    }
    
    if (!is.null(test.mask)) {
      fwrite(dffm[test.mask ], ffm.test.fn , sep = ' ', col.names = F)
    }
  }
  
  cat(date(), 'Preparing data in FFM format\n')
  dffm = ds.to.libffm(dat, labels)
  
  # We no longer need the full data, just enough to validate
  dat = dat[, .(display_id, ad_id)]; gc()
  # NOTE: this is to save memory, but it means I can't run XGB after FFM CV without reloading the data in between.
  
  if ('cv' %in% ffm.run.select) {
    cat(date(), 'Running FFM CV\n')
    
    train.disp.ids = unique(dat[train.mask, display_id])
    nr.folds = 3
    cv.folds = createFolds(train.disp.ids, k = nr.folds) # FIXME this random sampling is different from the "half random, half time split" used for the train/test split => validation performance here will be optimistic. I could try to create folds that have a time separation element (though not causal obviously)
    stacked.preds = rep(NA, nrow(dat))
    
    for (i in 1:nr.folds) {
      cat(date(), 'Fold', i, 'prep\n')
      
      fold.valid.mask = dat$display_id %in% train.disp.ids[cv.folds[[i]]]
      fold.train.mask = train.mask & !fold.valid.mask
      
      write.libffm(dffm, fold.train.mask, fold.valid.mask, NULL, ffm.train.fn, ffm.valid.fn, ffm.test.fn)
      gc()
      
      cat(date(), 'Fold', i, 'train\n')
      system(paste('./ffm-train -l', ffm.l, '-k', ffm.k, '-r', ffm.r, '-t', ffm.t.max, '-s', ffm.s, '-p', ffm.valid.fn, '--auto-stop', ffm.train.fn, ffm.model.fn))
      
      cat(date(), 'Fold', i, 'predict\n')
      system(paste('./ffm-predict', ffm.valid.fn, ffm.model.fn, ffm.preds.fn))
      
      stacked.preds[fold.valid.mask] = fread(ffm.preds.fn)$V1
    }
    
    if (valid.mode) {
      cat(date(), 'Writing train and valid sets to disk\n')
      write.libffm(dffm, train.mask, valid.mask, NULL, ffm.train.fn, ffm.valid.fn, ffm.test.fn)
      gc()
      
      cat(date(), 'Entire trainset train with validation early stop\n')
      system(paste('./ffm-train -l', ffm.l, '-k', ffm.k, '-r', ffm.r, '-t', ffm.t.max, '-s', ffm.s, '-p', ffm.valid.fn, '--auto-stop', ffm.train.fn, ffm.model.fn))
      
      cat(date(), 'Entire trainset predict on validation set\n')
      system(paste('./ffm-predict', ffm.valid.fn, ffm.model.fn, ffm.preds.fn))
      
      valid.preds = fread(ffm.preds.fn)$V1
      validate(dat, labels, valid.mask, valid.preds)
      
      cat(date(), 'Saving stacked predictions (validation version) to disk\n')
      stacked.preds[valid.mask] = valid.preds
      save(stacked.preds, file = ffm.stackedv.fn)
    } else {
      cat(date(), 'Writing train and test sets to disk\n')
      write.libffm(dffm, !test.mask, NULL, test.mask, ffm.train.fn, ffm.valid.fn, ffm.test.fn)
      gc()
      
      cat(date(), 'Entire trainset train with', ffm.t.final, 'iterations\n')
      system(paste('./ffm-train -l', ffm.l, '-k', ffm.k, '-r', ffm.r, '-t', ffm.t.final, '-s', ffm.s, ffm.train.fn, ffm.model.fn))
      
      cat(date(), 'Entire trainset predict on test set\n')
      system(paste('./ffm-predict', ffm.test.fn, ffm.model.fn, ffm.preds.fn))
      
      cat(date(), 'Saving stacked predictions (full version) to disk\n')
      stacked.preds[test.mask] = fread(ffm.preds.fn)$V1
      save(stacked.preds, file = ffm.stacked.fn)
    }
  }
  
  if ('valid' %in% ffm.run.select) {
    cat(date(), 'Running FFM valid\n')
    
    cat(date(), 'Writing FFM data files\n')
    write.libffm(dffm, train.mask, valid.mask, test.mask, ffm.train.fn, ffm.valid.fn, ffm.test.fn)
    
    cat(date(), 'Training FFM with early stopping on the validation set\n')
    system(paste('./ffm-train -l', ffm.l, '-k', ffm.k, '-r', ffm.r, '-t', ffm.t.max, '-s', ffm.s, '-p', ffm.valid.fn, '--auto-stop ', ffm.train.fn, ffm.model.fn))
    
    cat(date(), 'Generating validation predictions form FFM\n')  
    system(paste('./ffm-predict', ffm.valid.fn, ffm.model.fn, ffm.preds.fn))
    
    preds = fread(ffm.preds.fn)$V1
    validate(dat, labels, valid.mask, preds)
  }
  
  if ('final' %in% ffm.run.select) {
    cat(date(), 'Running FFM final\n')
    
    cat(date(), 'Writing FFM data files\n')
    write.libffm(dffm, !test.mask, NULL, test.mask, ffm.train.fn, ffm.valid.fn, ffm.test.fn)
    
    cat(date(), 'Training final FFM\n')
    system(paste('./ffm-train -l', ffm.l, '-k', ffm.k, '-r', ffm.r, '-t', ffm.t.final, '-s', ffm.s, ffm.train.fn, ffm.model.fn))
    
    cat(date(), 'Generating test predictions form FFM\n')  
    system(paste('./ffm-predict', ffm.test.fn, ffm.model.fn, ffm.preds.fn))
    preds = fread(ffm.preds.fn)$V1
  }
  
  rm(dffm); gc()
}

if (do.xgb) {
  xgb.ohe = F
  
  if (xgb.ohe) {
    xgb0.params = list(
      objective           = 'rank:map', 
      eval_metric         = 'map@12',
      maximize            = T,
     #objective           = 'reg:logistic', 
     #eval_metric         = 'logloss',
     #maximize            = F,
      booster             = 'gbtree',
      nrounds             = 1000,
      eta                 = 0.15,
     #alpha               = 1,
     #lambda              = 20,
      max_depth           = 10,
      subsample           = 1,
      colsample_bytree    = 0.3,
     #gamma               = 1,
     #lambda_bias         = 1,
      min_child_weight    = 6,
     #num_parallel_tree   = 5,
     #scale_pos_weight    = sum(train.labels == 0) / sum(train.labels == 1),
     #max_delta_step      = 1,
      annoying = T
    )
  } else {
    xgb0.params = list(
      objective           = 'rank:map', 
      eval_metric         = 'map@12',
      maximize            = T,
     #objective           = 'reg:logistic', 
     #eval_metric         = 'logloss',
     #maximize            = F,
      booster             = 'gbtree',
      nrounds             = 1000,
      eta                 = 0.1, #0.15,
     #alpha               = 1,
     #lambda              = 20,
      max_depth           = 8,
      subsample           = 1,
      colsample_bytree    = 0.5,
     #gamma               = 1,
     #lambda_bias         = 1,
      min_child_weight    = 12,
     #num_parallel_tree   = 5,
     #scale_pos_weight    = sum(train.labels == 0) / sum(train.labels == 1),
     #max_delta_step      = 1,
      annoying = T
    )
  }
  
  final.model.nrounds = 600 # FIXME make sure this matches the latest tuning runs
  
  xgb1.params = list(
    objective           = 'rank:map', 
    eval_metric         = 'map@12',
    maximize            = T,
   #objective           = 'reg:logistic', 
   #eval_metric         = 'logloss',
   #maximize            = F,
    booster             = 'gblinear',
    nrounds             = 1000,
    eta                 = 0.8, #0.3,
   #alpha               = 1,
    lambda              = 20,
    annoying = T
  )
  
  encode.main.ohe = function(x, xname) {
    # We will code all categories except the first one, and add an extra category for NA (if present)
    nr.levels = nlevels(x) - 1 + anyNA(x)
    ridx = seq_len(length(x))
    cidx = as.integer(x) - 1 # so in [0, ... nlevels(x) - 1] and maybe NA
    cidx[is.na(cidx)] = nr.levels
    mask = (cidx != 0)
    tmp = sparseMatrix(ridx[mask], cidx[mask], dims = c(length(x), nr.levels), check = F)
    colnames(tmp) = paste0(xname, '.', 1:ncol(tmp))
    return (tmp)
  }
  
  # Hmm... ok so this is extremely problematic because the number of nonzero elements in the full data is way higher 
  # than MAXINT, which is the maximum number supported by the Matrix package...
  #
  # So... I could try looking into lightgbm now instead of xgboost, maybe it can handle this case, or I could start
  # simplifying. For example: someone mentioned on the forum that in XGB he just left the categoricals in their raw 
  # numerical representation and got a very high score.
  #
  # !!!!!!! FIXME 
  # - Maybe with the treat-all-as-quantiative setup I need many more basal stacked and frequency coded 
  #   factors (since I don't have the OHE)
  # - Another approach is to use multiple different random level assignments where the levels are not obviously 
  #   meaningful (e.g. country?)
  # - Plus it makes sense that I'd have to re-tune...
  
  dt.to.dm = function(dat) {
    # This version just treats everything as quantitative, and leaves any NA as is
    # NOTE: it works in place within dat
    
    if (1) {
      if (valid.mode) {
        load(ffm.stackedv.fn) # => stacked.preds
        stacked.preds[is.na(stacked.preds)] = mean(stacked.preds, na.rm = T) # test samples aren't used so no matter
      } else {
        load(ffm.stacked.fn) # => stacked.preds
      }
      dat[, stacked := stacked.preds]
      rm(stacked.preds)
      
      # This seems to work better: within-display relative version (finally, some reasonable behavior!)
      dat[, stacked := stacked / max(stacked), by = display_id]
    }
    
    non.numerics = names(dat)[!sapply(dat, function(x) (class(x)[1] %in% c('integer', 'numeric')))]
    for (col in non.numerics) set(dat, j = col, value = as.integer(dat[[col]]))
    
    feat.names = setdiff(names(dat), c('display_id', 'ad_id'))
    
    return (feat.names)
  }
  
  dt.to.sm = function(dat) {
    # NOTE: Since this code is so inefficient, I have to canibalize dat
    
    # Aint using these for now
    dat[, (c('uuid', 'ad_target_subdomain', 'campaign_id', 'geo_location', 'uuid_top', 'country', 'state', 'time.utc', 'ad_id2')) := NULL]; gc()
    
    dat[is.na(disp_ad_pub_atd  ), disp_ad_pub_atd   := mean(dat$disp_ad_pub_atd  , na.rm = T)]
    dat[is.na(event_ad_pub_atd ), event_ad_pub_atd  := mean(dat$event_ad_pub_atd , na.rm = T)]
    
    # dat[, day   := yday(time) - 165] # will not generalize to test-only dates
    # dat[, wday  := wday(time)]
    # dat[, hour1 := hour(time)]
    # dat[, hour2 := (hour(time) - 12) %% 24]
    # dat[, time  := NULL]
    
    if (1) {
      if (valid.mode) {
        load(ffm.stackedv.fn) # => stacked.preds
        stacked.preds[is.na(stacked.preds)] = mean(stacked.preds, na.rm = T)
      } else {
        load(ffm.stacked.fn) # => stacked.preds
      }
      dat[, stacked := stacked.preds]
      rm(stacked.preds)
    }
    
    if (0) {
      # A within-display-rank of a marginally strong feature
      tmp[, c('idx', 'display_id') := .(1:.N, dat$display_id)]
      setorder(tmp, display_id, stacked)
      tmp[, strong.feature.wdr := 1:.N, by = display_id]
      setorder(tmp, idx)
      tmp[, c('idx', 'display_id') := NULL]
    }
    
    # FIXME: I had to drop some of the dense (quantitative, no dominating single value) features below to save memory
   #mat = Matrix(data.matrix(dat[, .(ad_target_popularity, disp_ad_pub_atd, event_ad_pub_atd)]), sparse = T)
    dat[, (c('ad_target_popularity', 'disp_ad_pub_atd', 'event_ad_pub_atd')) := NULL]; gc()
    mat = cbind(mat, data.matrix(dat[, .(leak, uuid_campaign, uuid_domain, uuid_subdomain, uuid_cat, uuid_ent)]))
    dat[, (c('leak', 'uuid_campaign', 'uuid_domain', 'uuid_subdomain', 'uuid_cat', 'uuid_ent')) := NULL]; gc()
   #mat = cbind(mat, data.matrix(dat[, .(ad.evcnt, ad.target.evcnt, campaign_id.evcnt, advertiser_id.evcnt, country.camp.evcnt)]))
    dat[, (c('ad.evcnt', 'ad.target.evcnt', 'campaign_id.evcnt', 'advertiser_id.evcnt', 'country.camp.evcnt')) := NULL]; gc()
    mat = cbind(mat, data.matrix(dat[, .(state.meta, ad.target.meta, campaign_id.meta, advertiser_id.meta, ad.meta, ad.targetXplatform.meta, ad.targetXcountry.meta)]))
    dat[, (c('state.meta', 'ad.target.meta', 'campaign_id.meta', 'advertiser_id.meta', 'ad.meta', 'ad.targetXplatform.meta', 'ad.targetXcountry.meta')) := NULL]; gc()
    
    # These binary features are coded as categorical in order to properly capture missing values
    mat = cbind(mat, encode.main.ohe(as.factor(dat$disp_ad_pub_std  ), 'disp_ad_pub_std'  ))
    mat = cbind(mat, encode.main.ohe(as.factor(dat$event_ad_pub_std ), 'event_ad_pub_std' ))
    mat = cbind(mat, encode.main.ohe(as.factor(dat$disp_ad_cat_match), 'disp_ad_cat_match'))
    mat = cbind(mat, encode.main.ohe(as.factor(dat$disp_ad_ent_match), 'disp_ad_ent_match'))
    mat = cbind(mat, encode.main.ohe(as.factor(dat$disp_ad_top_match), 'disp_ad_top_match'))
    dat[, (c('disp_ad_pub_std', 'event_ad_pub_std', 'disp_ad_cat_match', 'disp_ad_ent_match', 'disp_ad_top_match')) := NULL]; gc()
    
    if (1) {
      # FIXME since this is hierarchical need to eliminate levels (for gblinear anyway)
      mat = cbind(mat, encode.main.ohe(dat$advertiser_id      , 'advertiser_id'      ))
     #mat = cbind(mat, encode.main.ohe(dat$campaign_id        , 'campaign_id'        )) # NOTE: fragmented
      mat = cbind(mat, encode.main.ohe(dat$ad_target          , 'ad_target'          ))
     #mat = cbind(mat, encode.main.ohe(dat$ad_id2             , 'ad_id'              )) # NOTE: fragmanted
      mat = cbind(mat, encode.main.ohe(dat$ad_target_domain   , 'ad_target_domain'   ))
     #mat = cbind(mat, encode.main.ohe(dat$ad_target_subdomain, 'ad_target_subdomain'))
      mat = cbind(mat, encode.main.ohe(dat$ad_target_cat      , 'ad_target_cat'      ))
      mat = cbind(mat, encode.main.ohe(dat$ad_target_ent      , 'ad_target_ent'      )) # NOTE: fragmanted
      mat = cbind(mat, encode.main.ohe(dat$ad_target_top      , 'ad_target_top'      ))
      dat[, (c('advertiser_id', 'ad_target', 'ad_target_domain', 'ad_target_cat', 'ad_target_ent', 'ad_target_top')) := NULL]; gc()
    }
    
    if (1) {
      # FIXME these are also nested so redundant and colinear..
     #mat = cbind(mat, encode.main.ohe(dat$uuid               , 'uuid'               ))
     #mat = cbind(mat, encode.main.ohe(dat$country            , 'country'            ))
     #mat = cbind(mat, encode.main.ohe(dat$state              , 'state'              )) # maybe
     #mat = cbind(mat, encode.main.ohe(dat$geo_location       , 'geo_location'       )) # NOTE: fragmented
      mat = cbind(mat, encode.main.ohe(dat$platform           , 'platform'           ))
      
      mat = cbind(mat, encode.main.ohe(dat$disp_doc_domain    , 'disp_doc_domain'    ))
      mat = cbind(mat, encode.main.ohe(dat$disp_doc_subdomain , 'disp_doc_subdomain' ))
      mat = cbind(mat, encode.main.ohe(dat$disp_doc_cat       , 'disp_doc_cat'       ))
      mat = cbind(mat, encode.main.ohe(dat$disp_doc_ent       , 'disp_doc_ent'       ))
      mat = cbind(mat, encode.main.ohe(dat$disp_doc_top       , 'disp_doc_top'       ))
      mat = cbind(mat, encode.main.ohe(dat$disp_docid         , 'disp_docid'         ))
      
      dat[, (c('platform', 'disp_doc_domain', 'disp_doc_subdomain', 'disp_doc_cat', 'disp_doc_ent', 'disp_doc_top', 'disp_docid')) := NULL]; gc()
    }
    
    # For gblinear, we might want to add some interactions:
    # TODO many more interactions...
    
    if (0) { # Interaction ad_id X "out of office"
      ridx = seq_len(nrow(dat))
      cidx = as.integer(dat$ad_id2)
      cidx[is.na(cidx)] = nlevels(dat$ad_id2) + 1
      mask = (dat$is.office.hour == 0)
      tmp = sparseMatrix(ridx[mask], cidx[mask], dims = c(nrow(dat), nlevels(dat$ad_id2) + 1))
      colnames(tmp) = paste0('ad_id__ooo', 1:ncol(tmp))
      mat = cbind(mat, tmp)
      rm(tmp); gc()
    }
    
    if (0) { # Interaction ad_id X platform
      # There are a handful of NAs, then many 1s, 2s and 3s. I'll merge the 0s into 1s 
      # (arbitrarily), use this as the reference level, and add contrasts for 2s and 3s 
      platform = as.integer(dat$platform)
      platform[is.na(platform)] = 0
      ridx = seq_len(nrow(dat))
      cidx = as.integer(dat$ad_id2) - 1
      cidx[is.na(cidx)] = nlevels(dat$ad_id2)
      cidx = 1 + 2 * cidx + (platform == '3')
      mask = (platform %in% c('2', '3'))
      tmp = sparseMatrix(ridx[mask], cidx[mask], dims = c(nrow(dat), 2 * (nlevels(dat$ad_id2) + 1)))
      colnames(tmp) = paste0('ad_id__time.', 1:ncol(tmp))
      mat = cbind(mat, tmp)
      rm(tmp); gc()
    }
    
    return (mat)
  }
  
  if (!xgbd.from.disk) {
    cat(date(), 'Preparing data for XGB\n')
    if (xgb.ohe) {
      sdat = dt.to.sm(dat)
    } else {
      feat.names = dt.to.dm(dat)
    }
  }
  
  if ('stack' %in% xgb.run.select) {
    cat(date(), 'Running XGB stack\n')
    # Stacking gbtree->gblinear
    # NOTE: I tired to train a gbtree in L0 and gblinear in L1, and vice versa... it doesn't seem to improve over
    # gbtree alone.
    
    # FIXME fow now, this assumes validation mode, and ignoes the test set
    
    cat(date(), 'Training L0 XGB\n')
    
    train.disp.ids = unique(dat[train.mask, display_id])
    nr.folds = 3
    cv.folds = createFolds(train.disp.ids, k = nr.folds) # FIXME this random sampling is different from the "half random, half time split" used for the train/test split => validation performance here will be optimistic
    train.preds0 = rep(NA, nrow(dat))
    
    for (i in 1:nr.folds) {
      cat(date(), 'Fold', i, 'prep\n')
      
      fold.valid.mask = dat$display_id %in% train.disp.ids[cv.folds[[i]]]
      fold.train.mask = train.mask & !fold.valid.mask
      fold.xtrain = xgb.DMatrix(sdat[fold.train.mask, ], label = labels[fold.train.mask], group = dat[fold.train.mask, .N, by = display_id]$N)
      fold.xvalid = xgb.DMatrix(sdat[fold.valid.mask, ], label = labels[fold.valid.mask], group = dat[fold.valid.mask, .N, by = display_id]$N)
      #gc()
      
      cat(date(), 'Fold', i, 'train\n')
      xgb0 = xgb.train(
        early_stopping_rounds = 20,
        nrounds           = xgb0.params$nrounds,
        params            = xgb0.params,
        maximize          = xgb0.params$maximize,
        data              = fold.xtrain,
        watchlist         = list(train = fold.xtrain, valid = fold.xvalid), # FIXME it'll be optimistic since I early-stop on the same validset
        print_every_n     = 10
        #nthread           = 8
      )
      
      train.preds0[fold.valid.mask] = predict(xgb0, fold.xvalid, ntreelimit = 0)
      rm(fold.xtrain, fold.xvalid, fold.train.mask, fold.valid.mask); #gc()
    }
    
    train.preds0 = train.preds0[train.mask]
    
    xtrain = xgb.DMatrix(sdat[train.mask, ], label = labels[train.mask], group = dat[train.mask, .N, by = display_id]$N)
    xvalid = xgb.DMatrix(sdat[valid.mask, ], label = labels[valid.mask], group = dat[valid.mask, .N, by = display_id]$N)
    
    cat(date(), 'Final L0 XGB train\n')
    xgb0 = xgb.train(
      early_stopping_rounds = 20,
      nrounds           = xgb0.params$nrounds,
      params            = xgb0.params,
      maximize          = xgb0.params$maximize,
      data              = xtrain,
      watchlist         = list(train = xtrain, valid = xvalid), # FIXME it'll be optimistic since I early-stop on the same validset
      print_every_n     = 10
      #nthread           = 8
    )
    
    if (0) {
      impo = xgb.importance(feature_names = colnames(xtrain), xgb0)
      impo[, idx := 1:.N]
      impo[, relGain := Gain / Gain[1]]
      View(impo)
    }
    
    valid.preds0 = predict(xgb0, xvalid, ntreelimit = 0)
    validate(dat, labels, valid.mask, valid.preds0)
    rm(xvalid); #gc()
    
    # Save stacked XGB, it might be useful in other models
    xgb.meta = rep(NA, nrow(dat))
    xgb.meta[train.mask] = train.preds0
    xgb.meta[valid.mask] = valid.preds0
    if (debug.mode) {
      save(xgb.meta, file = paste0(tmpdir, '/xgb-meta-sample.RData'))
    } else {
      save(xgb.meta, file = paste0(tmpdir, '/xgb-meta.RData'))
    }
    
    cat(date(), 'Preparing data for L1 XGB\n')
    xtrain = xgb.DMatrix(cbind(sdat[train.mask, ], stacked = train.preds0), label = labels[train.mask], group = dat[train.mask, .N, by = display_id]$N)
    xvalid = xgb.DMatrix(cbind(sdat[valid.mask, ], stacked = valid.preds0), label = labels[valid.mask], group = dat[valid.mask, .N, by = display_id]$N)
    #gc()
    
    cat(date(), 'Training L1 XGB\n')
    
    xgb = xgb.train(
      early_stopping_rounds = 20,
      nrounds           = xgb1.params$nrounds,
      params            = xgb1.params,
      maximize          = xgb1.params$maximize,
      data              = xtrain,
      watchlist         = list(train = xtrain, valid = xvalid), # FIXME it'll be optimistic since I early-stop on the same validset
      print_every_n     = 10
      #nthread           = 8
    )
    
    preds = predict(xgb, xvalid, ntreelimit = 0)
    rm(xtrain, xvalid); #gc()
    
    validate(dat, labels, valid.mask, preds)
    gc()
  } 
  
  if ('valid' %in% xgb.run.select) {
    cat(date(), 'Running XGB valid\n')
    stopifnot(valid.mode)
    
    gc()
    if (xgb.ohe) {
      xtrain = xgb.DMatrix(sdat[train.mask, ], label = labels[train.mask], group = dat[train.mask, .N, by = display_id]$N)
      xvalid = xgb.DMatrix(sdat[valid.mask, ], label = labels[valid.mask], group = dat[valid.mask, .N, by = display_id]$N)
    } else {
      if (!xgbd.from.disk) {
        #xtrain = xgb.DMatrix(data.matrix(dat[train.mask, feat.names, with = F]), label = labels[train.mask], group = dat[train.mask, .N, by = display_id]$N, missing = NA); gc()
        #xvalid = xgb.DMatrix(data.matrix(dat[valid.mask, feat.names, with = F]), label = labels[valid.mask], group = dat[valid.mask, .N, by = display_id]$N, missing = NA); gc()
        
        cat(date(), 'NOTE: saving XGB matrices to disk\n')
        valid.groups = dat[valid.mask, .N, by = display_id]$N
        train.groups = dat[train.mask, .N, by = display_id]$N
        gc()
        dvalid = data.matrix(dat[valid.mask, feat.names, with = F])
        dat = dat[train.mask, feat.names, with = F]
        gc()
        xvalid = xgb.DMatrix(dvalid, label = labels[valid.mask], group = valid.groups, missing = NA)
        rm(dvalid, valid.groups); gc()
        xgb.DMatrix.save(xvalid, paste0(tmpdir, '/xvalid.xgbd'))
        dtrain = data.matrix(dat)
        rm(dat); gc()
        xtrain = xgb.DMatrix(dtrain, label = labels[train.mask], group = train.groups, missing = NA); 
        xgb.DMatrix.save(xtrain, paste0(tmpdir, '/xtrain.xgbd'))
      } else {
        cat(date(), 'NOTE: Loading XGB matrices from disk\n')
        xtrain = xgb.DMatrix(paste0(tmpdir, '/xtrain.xgbd'))
        xvalid = xgb.DMatrix(paste0(tmpdir, '/xvalid.xgbd'))
      }
    }
    
    cat(date(), 'Training XGB\n')
    xgb = xgb.train(
      early_stopping_rounds = 100,
      nrounds           = xgb0.params$nrounds,
      params            = xgb0.params,
      maximize          = xgb0.params$maximize,
      data              = xtrain,
      watchlist         = list(train = xtrain, valid = xvalid), # FIXME it'll be optimistic since I early-stop on the same validset
      print_every_n     = 1,
      save_period       = 10
      #nthread           = 8
    )
    
    if (1) {
      impo = xgb.importance(feature_names = colnames(xtrain), xgb)
      impo[, idx := 1:.N]
      impo[, relGain := Gain / Gain[1]]
      View(impo)
    }
    
    valid.preds = predict(xgb, xvalid, ntreelimit = 0)
    validate(dat, labels, valid.mask, valid.preds)
    rm(xtrain, xvalid); gc()
  }
  
  if ('final' %in% xgb.run.select) {
    cat(date(), 'Running XGB final\n')
    
    if (xgb.ohe) {
      xtrain = xgb.DMatrix(sdat[!test.mask, ], label = labels[!test.mask], group = dat[!test.mask, .N, by = display_id]$N); gc()
    } else {
      if (!xgbd.from.disk) {
        # RUN THIS ON A FAST MACHINE WITH A TON OF MEM (150GB?)
        rm(train.mask)
        train.groups = dat[!test.mask, .N, by = display_id]$N
        test.groups  = dat[ test.mask, .N, by = display_id]$N
        dat[, (c('display_id', 'ad_id')) := NULL]
        gc()
        dtrain = data.matrix(dat[!test.mask])
        dat = dat[test.mask]
        gc()
        xtrain = xgb.DMatrix(dtrain, label = labels[!test.mask], group = train.groups, missing = NA)
        rm(dtrain, train.groups, labels); gc()
        xgb.DMatrix.save(xtrain, paste0(tmpdir, '/xtrain.xgbd'))
        rm(xtrain); gc()
        dtest = data.matrix(dat)
        rm(dat); gc()
        xtest = xgb.DMatrix(dtest, group = test.groups, missing = NA); 
        xgb.DMatrix.save(xtest, paste0(tmpdir, '/xtest.xgbd'))
        rm(xtest); gc()
      }
      
      xtrain = xgb.DMatrix(paste0(tmpdir, '/xtrain.xgbd'))
    }
    
    cat(date(), 'Training XGB\n')
    xgb = xgb.train(
      nrounds           = final.model.nrounds,
      params            = xgb0.params,
      maximize          = xgb0.params$maximize,
      data              = xtrain,
      watchlist         = list(train = xtrain),
      print_every_n     = 10,
      save_period       = 10
      #nthread           = 8
    )
    
    rm(xtrain); gc()
    if (xgb.ohe) {
      xtest = xgb.DMatrix(sdat[test.mask , ], group = dat[test.mask, .N, by = display_id]$N); gc()
    } else {
      xtest = xgb.DMatrix(paste0(tmpdir, '/xtest.xgbd'))
    }
    
    preds = predict(xgb, xtest, ntreelimit = 0)
    save(preds, file = paste0(tmpdir, '/test-preds.RData'))
    rm(xtest); gc()
  }
}

if (do.lgb) {
  # NOTE: The code here is mostly just duplicayed  from XGB for lack of time, oops.
  
  lgb.params = list(
    objective        = 'lambdarank', # hmm unfortunately it only supports NDCG (no MAP support for now) so it won't be exactly ERM
    max_position     = 12,
    ndcg_at          = 12,
    label_gain       = 1:12, # I'm not sure how to make this the most similar to MAP
    num_iterations   = 1000,
    learning_rate    = 0.1,
    max_depth        = 8,
    min_data_in_leaf = 12,
    feature_fraction = 0.5,
   #num_leaves       = 63,
   #min_sum_hessian_in_leaf = 12,
   #max_bin          = 255
    annoying = T
  )
  
  dt.to.dm = function(dat) {
    if (valid.mode) {
      load(ffm.stackedv.fn) # => stacked.preds
      stacked.preds[is.na(stacked.preds)] = mean(stacked.preds, na.rm = T) # test samples aren't used so no matter
    } else {
      load(ffm.stacked.fn) # => stacked.preds
    }
    dat[, stacked := stacked.preds]
    rm(stacked.preds)
    
    dat[, stacked := stacked / max(stacked), by = display_id]
    
    non.numerics = names(dat)[!sapply(dat, function(x) (class(x)[1] %in% c('integer', 'numeric')))]
    for (col in non.numerics) set(dat, j = col, value = as.integer(dat[[col]]))
    
    feat.names = setdiff(names(dat), c('display_id', 'ad_id'))
    
    return (feat.names)
  }
  
  if (!xgbd.from.disk) {
    cat(date(), 'Preparing data for LGB\n')
    feat.names = dt.to.dm(dat)
  }
  
  if ('valid' %in% xgb.run.select) {
    cat(date(), 'Running LGB valid\n')
    stopifnot(valid.mode)
    
    gc()
    if (!xgbd.from.disk) {
      #xtrain = lgb.Dataset(data.matrix(dat[train.mask, feat.names, with = F]), label = labels[train.mask], group = dat[train.mask, .N, by = display_id]$N, missing = NA); gc()
      #xvalid = lgb.Dataset(data.matrix(dat[valid.mask, feat.names, with = F]), label = labels[valid.mask], group = dat[valid.mask, .N, by = display_id]$N, missing = NA); gc()
      
      cat(date(), 'NOTE: saving LGB matrices to disk\n')
      valid.groups = dat[valid.mask, .N, by = display_id]$N
      train.groups = dat[train.mask, .N, by = display_id]$N
      gc()
      dvalid = data.matrix(dat[valid.mask, feat.names, with = F])
      dat.legacy = dat[, .(display_id, ad_id)] # we have to compute our own MAP
      dat = dat[train.mask, feat.names, with = F]
      gc()
      xvalid = lgb.Dataset(dvalid, label = labels[valid.mask], group = valid.groups, missing = NA, free_raw_data = F)
      unlink(paste0(tmpdir, '/valid.lgbd')) # it won't write otherwise.. strange
      lgb.Dataset.save(xvalid, paste0(tmpdir, '/valid.lgbd'))
      #rm(dvalid) # predict needs it.. can't work with anything else yet it seems
      rm(valid.groups); gc()
      dtrain = data.matrix(dat)
      rm(dat); gc()
      xtrain = lgb.Dataset(dtrain, label = labels[train.mask], group = train.groups, missing = NA, free_raw_data = F)
      unlink(paste0(tmpdir, '/train.lgbd')) # it won't write otherwise.. strange
      lgb.Dataset.save(xtrain, paste0(tmpdir, '/train.lgbd'))
      rm(dtrain); gc()
    } else {
      cat(date(), 'NOTE: Loading LGB matrices from disk\n')
      xtrain = lgb.Dataset(paste0(tmpdir, '/train.lgbd'))
      xvalid = lgb.Dataset(paste0(tmpdir, '/valid.lgbd'))
    }
    
    cat(date(), 'Training LGB\n')
    lgb = lgb.train(
      #init_model = lgb, # for incremental training
      early_stopping_rounds = 100,
      nrounds           = lgb.params$num_iterations,
      params            = lgb.params,
      data              = xtrain,
      valids            = list(train = xtrain, valid = xvalid), # FIXME it'll be optimistic since I early-stop on the same validset
      eval_freq         = 10
     #save_period       = 10 # not available yet
     #nthread           = 8
    )
    
    # (there is no importance functionality yet)    
    
    valid.preds = predict(lgb, dvalid) # doesn't work yet on file or xvalid
    validate(dat.legacy, labels, valid.mask, valid.preds)
    rm(xtrain, xvalid, lgb); gc()
  }
  
  if ('final' %in% xgb.run.select) {
    cat(date(), 'Running LGB final\n')
    
    if (!xgbd.from.disk) {
      rm(train.mask)
      train.groups = dat[!test.mask, .N, by = display_id]$N
      test.groups  = dat[ test.mask, .N, by = display_id]$N
      dat[, (c('display_id', 'ad_id')) := NULL]
      gc()
      dtrain = data.matrix(dat[!test.mask])
      dat = dat[test.mask]
      gc()
      xtrain = lgb.Dataset(dtrain, label = labels[!test.mask], group = train.groups, missing = NA, free_raw_data = F)
      rm(dtrain, train.groups, labels); gc()
      unlink(paste0(tmpdir, '/train.lgbd')) # it won't write otherwise.. strange
      lgb.Dataset.save(xtrain, paste0(tmpdir, '/train.lgbd'))
      rm(xtrain); gc()
      dtest = data.matrix(dat)
      rm(dat); gc()
      xtest = lgb.Dataset(dtest, group = test.groups, missing = NA, free_raw_data = F)
      unlink(paste0(tmpdir, '/test.lgbd')) # it won't write otherwise.. strange
      lgb.Dataset.save(xtest, paste0(tmpdir, '/test.lgbd'))
    } else {
      xtrain = lgb.Dataset(paste0(tmpdir, '/train.lgbd'))
    }
    
    cat(date(), 'Training LGB\n')
    lgb = lgb.train(
      nrounds           = lgb.params$num_iterations,
      params            = lgb.params,
      data              = xtrain,
      valids            = list(train = xtrain),
      eval_freq         = 1
     #save_period       = 10 # not available yet
     #nthread           = 8
    )
    
    rm(xtrain, xtest); gc()
    preds = predict(lgb, dtest)
    save(preds, file = paste0(tmpdir, '/test-preds2.RData'))
    
    if (1) {
      # Blend with the best XGB submission
      preds2 = preds
      load(paste0(tmpdir, '/test-preds.RData'))
      ppp = data.table(preds, preds2)
      preds = (scale(preds) + scale(preds2)) / 2
    }
  }
}

if (do.submit) {
  cat(date(), 'Generating submission\n')
  generate.submission(submission.id, preds)
}

cat(date(), 'Done.\n')

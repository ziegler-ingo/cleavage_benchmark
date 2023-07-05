import sys
from tqdm import tqdm

import torch

from sklearn.metrics import roc_auc_score, confusion_matrix

from denoise import NoiseAdaptation
from utils import (
    save_metrics_base,
    save_metrics_coteaching,
    save_metrics_jocor,
    save_metrics_nad,
    regularized_auc,
)


def run_epochs_base(
    model,
    model_type,
    train_loader,
    val_loader,
    criterion,
    device,
    fold,
    num_epochs,
    highest_val_auc,
    early_stop,
    logging_path,
    param_path,
    optim=None,
    scaler=None,
):
    highest_val_auc = highest_val_auc
    num_overfitted, num_neg_progress = 0, 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, train_acc, train_auc = train_or_eval_base(
            model,
            model_type,
            train_loader,
            criterion,
            device,
            optim=optim,
            scaler=scaler,
        )

        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_auc = train_or_eval_base(
                model, model_type, val_loader, criterion, device, scaler=scaler
            )

        # save metrics
        save_metrics_base(
            fold,
            epoch,
            train_loss,
            train_acc,
            train_auc,
            val_loss,
            val_acc,
            val_auc,
            path=logging_path,
        )

        print(
            f"Training:   [Fold {fold:2d}, Epoch {epoch:2d}, Loss: {train_loss:8.6f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}]"
        )
        print(
            f"Evaluation: [Fold {fold:2d}, Epoch {epoch:2d}, Loss: {val_loss:8.6f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}]"
        )

        reg_auc = regularized_auc(train_auc, val_auc, threshold=0)
        num_overfitted = num_overfitted + 1 if not reg_auc else 0
        num_neg_progress = num_neg_progress + 1 if val_auc < highest_val_auc else 0
        _path = param_path + f"auc{reg_auc:.4f}_fold{fold}_epoch{epoch}.pt"
        if num_overfitted == early_stop or num_neg_progress == early_stop:
            print(f"Early stopped after epoch {epoch}.")
            print(
                f"num_overfitted={num_overfitted}, num_neg_progress={num_neg_progress}"
            )
        if fold == 1 and epoch == 1:
            # save first model in all cases
            torch.save(model.state_dict(), _path)
        elif reg_auc > highest_val_auc:
            highest_val_auc = reg_auc
            torch.save(model.state_dict(), _path)

    return highest_val_auc


def run_epochs_coteach(
    model_type,
    train_loader,
    val_loader,
    model1,
    model2,
    rate_schedule,
    device,
    fold,
    num_epochs,
    highest_val_auc,
    early_stop,
    logging_path,
    param_path,
    cot_criterion=None,
    criterion=None,
    optim1=None,
    optim2=None,
    scaler1=None,
    scaler2=None,
    cot_plus_train=None,
):
    highest_val_auc = highest_val_auc
    num_overfitted, num_neg_progress = 0, 0
    for epoch in range(1, num_epochs + 1):
        model1.train()
        model2.train()
        train_res = train_or_eval_coteaching(
            model_type=model_type,
            loader=train_loader,
            model1=model1,
            model2=model2,
            device=device,
            forget_rate=rate_schedule[epoch - 1],
            cot_criterion=cot_criterion,
            criterion=criterion if cot_plus_train is not None else None,
            optim1=optim1,
            optim2=optim2,
            scaler1=scaler1,
            scaler2=scaler2,
            cot_plus_train=cot_plus_train,
        )

        model1.eval()
        model2.eval()
        with torch.no_grad():
            val_res = train_or_eval_coteaching(
                model_type=model_type,
                loader=val_loader,
                model1=model1,
                model2=model2,
                device=device,
                criterion=criterion,
                scaler1=scaler1,
                scaler2=scaler2,
            )

        # save metrics
        save_metrics_coteaching(
            fold,
            epoch,
            train_res[0],  # train_loss1
            train_res[1],  # train_loss2
            train_res[2],  # train_acc1
            train_res[3],  # train_acc2
            train_res[4],  # train_auc1
            train_res[5],  # train_auc2
            val_res[0],  # val_loss1
            val_res[1],  # val_loss2
            val_res[2],  # val_acc1
            val_res[3],  # val_acc2
            val_res[4],  # val_auc1
            val_res[5],  # val_auc2
            path=logging_path,
        )

        print(
            f"Training1:   [Fold {fold:2d}, Epoch {epoch:2d}, Loss: {train_res[0]:8.6f}, Acc: {train_res[2]:.4f}, AUC: {train_res[4]:.4f}]"
        )
        print(
            f"Training2:   [Fold {fold:2d}, Epoch {epoch:2d}, Loss: {train_res[1]:8.6f}, Acc: {train_res[3]:.4f}, AUC: {train_res[5]:.4f}]"
        )
        print(
            f"Evaluation1: [Fold {fold:2d}, Epoch {epoch:2d}, Loss: {val_res[0]:8.6f}, Acc: {val_res[2]:.4f}, AUC: {val_res[4]:.4f}]"
        )
        print(
            f"Evaluation2: [Fold {fold:2d}, Epoch {epoch:2d}, Loss: {val_res[1]:8.6f}, Acc: {val_res[3]:.4f}, AUC: {val_res[5]:.4f}]"
        )

        reg_auc1 = regularized_auc(train_res[4], val_res[4], threshold=0)
        reg_auc2 = regularized_auc(train_res[5], val_res[5], threshold=0)
        # if both models underperform at the same time, start counting towards early stopping
        # if one model is still increasing performance, continue training
        num_overfitted = num_overfitted + 1 if not reg_auc1 and not reg_auc2 else 0
        num_neg_progress = (
            num_neg_progress + 1
            if val_res[4] < highest_val_auc and val_res[5] < highest_val_auc
            else 0
        )
        if num_overfitted == early_stop or num_neg_progress == early_stop:
            print(f"Early stopped after epoch {epoch}.")
            print(
                f"num_overfitted={num_overfitted}, num_neg_progress={num_neg_progress}"
            )

        if reg_auc1 > reg_auc2:
            _reg_auc = reg_auc1
            _path = param_path + f"auc{reg_auc1:.4f}_fold{fold}_epoch{epoch}.pt"
            _model = model1.state_dict()
            print("Saved first model")
        else:
            _reg_auc = reg_auc2
            _path = param_path + f"auc{reg_auc2:.4f}_fold{fold}_epoch{epoch}.pt"
            _model = model2.state_dict()
            print("Saved second model")
        if fold == 1 and epoch == 1:
            # save first model in all cases
            torch.save(_model, _path)
        elif _reg_auc > highest_val_auc:
            highest_val_auc = _reg_auc
            torch.save(_model, _path)

    return highest_val_auc


def run_epochs_jocor(
    model_type,
    train_loader,
    val_loader,
    model1,
    model2,
    rate_schedule,
    device,
    fold,
    num_epochs,
    highest_val_auc,
    early_stop,
    logging_path,
    param_path,
    jocor_criterion=None,
    optim=None,
    scaler=None,
):
    highest_val_auc = highest_val_auc
    num_overfitted, num_neg_progress = 0, 0
    for epoch in range(1, num_epochs + 1):
        model1.train()
        model2.train()
        train_res = train_or_eval_jocor(
            model_type=model_type,
            loader=train_loader,
            model1=model1,
            model2=model2,
            device=device,
            forget_rate=rate_schedule[epoch - 1],
            jocor_criterion=jocor_criterion,
            optim=optim,
            scaler=scaler,
        )

        model1.eval()
        model2.eval()
        with torch.no_grad():
            val_res = train_or_eval_jocor(
                model_type=model_type,
                loader=val_loader,
                model1=model1,
                model2=model2,
                device=device,
                scaler=scaler,
            )

        # save metrics
        save_metrics_jocor(
            fold,
            epoch,
            train_res[0],  # train_loss
            train_res[1],  # train_acc1
            train_res[2],  # train_acc2
            train_res[3],  # train_auc1
            train_res[4],  # train_auc2
            val_res[0],  # val_loss, placeholder value
            val_res[1],  # val_acc1
            val_res[2],  # val_acc2
            val_res[3],  # val_auc1
            val_res[4],  # val_auc2
            path=logging_path,
        )

        print(
            f"Training1:   [Fold {fold:2d}, Epoch {epoch:2d}, Loss: {train_res[0]:8.6f}, Acc: {train_res[1]:.4f}, AUC: {train_res[3]:.4f}]"
        )
        print(
            f"Training2:   [Fold {fold:2d}, Epoch {epoch:2d}, Loss: {train_res[0]:8.6f}, Acc: {train_res[2]:.4f}, AUC: {train_res[4]:.4f}]"
        )
        print(
            f"Evaluation1: [Fold {fold:2d}, Epoch {epoch:2d}, Loss: {val_res[0]:8.6f}, Acc: {val_res[1]:.4f}, AUC: {val_res[3]:.4f}]"
        )
        print(
            f"Evaluation2: [Fold {fold:2d}, Epoch {epoch:2d}, Loss: {val_res[0]:8.6f}, Acc: {val_res[2]:.4f}, AUC: {val_res[4]:.4f}]"
        )

        reg_auc1 = regularized_auc(train_res[3], val_res[3], threshold=0)
        reg_auc2 = regularized_auc(train_res[4], val_res[4], threshold=0)
        # if both models underperform at the same time, start counting towards early stopping
        # if one model is still increasing performance, continue training
        num_overfitted = num_overfitted + 1 if not reg_auc1 and not reg_auc2 else 0
        num_neg_progress = (
            num_neg_progress + 1
            if val_res[3] < highest_val_auc and val_res[4] < highest_val_auc
            else 0
        )
        if num_overfitted == early_stop or num_neg_progress == early_stop:
            print(f"Early stopped after epoch {epoch}.")
            print(
                f"num_overfitted={num_overfitted}, num_neg_progress={num_neg_progress}"
            )

        if reg_auc1 > reg_auc2:
            _reg_auc = reg_auc1
            _path = param_path + f"auc{reg_auc1:.4f}_fold{fold}_epoch{epoch}.pt"
            _model = model1.state_dict()
            print("Saved first model")
        else:
            _reg_auc = reg_auc2
            _path = param_path + f"auc{reg_auc2:.4f}_fold{fold}_epoch{epoch}.pt"
            _model = model2.state_dict()
            print("Saved second model")
        if fold == 1 and epoch == 1:
            # save first model in all cases
            torch.save(_model, _path)
        elif _reg_auc > highest_val_auc:
            highest_val_auc = _reg_auc
            torch.save(_model, _path)

    return highest_val_auc


def run_epochs_nad(
    model_type,
    train_loader,
    val_loader,
    model,
    device,
    fold,
    num_epochs,
    num_warmup,
    learning_rate,
    beta,
    highest_val_auc,
    early_stop,
    logging_path,
    param_path,
    criterion=None,
    optim=None,
    scaler=None,
):
    highest_val_auc = highest_val_auc
    num_overfitted, num_neg_progress = 0, 0
    noise_model, noise_optim = None, None

    for epoch in range(1, num_epochs + 1):
        if epoch < num_warmup + 1:
            # run warmup epoch
            model.train()
            train_loss, train_acc, train_auc = train_or_eval_base(
                model,
                model_type,
                train_loader,
                criterion,
                device,
                optim=optim,
                scaler=scaler,
                nad=True,
            )

            model.eval()
            with torch.no_grad():
                val_loss, val_acc, val_auc = train_or_eval_base(
                    model,
                    model_type,
                    val_loader,
                    criterion,
                    device,
                    scaler=scaler,
                    nad=True,
                )

            print(
                f"Warmup Train: [Fold {fold:2d}, Epoch {epoch:2d}, Loss: {train_loss:8.6f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}]"
            )
            print(
                f"Warmup Eval: [Fold {fold:2d}, Epoch {epoch:2d}, Loss: {val_loss:8.6f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}]"
            )

            if epoch == num_warmup:
                # get conf matrix based on predictions on train data
                model.eval()
                conf_matrix = get_confusion_matrix(
                    model, model_type, train_loader, device, scaler=scaler
                )
                theta = conf_matrix / conf_matrix.sum(dim=1, keepdim=True)
                theta = torch.log(theta + 1e-8).float()  # avoid zeros with +1e-8

                # create noisemodel
                noise_model = NoiseAdaptation(theta=theta, k=2, device=device).to(
                    device
                )
                noise_optim = torch.optim.Adam(
                    noise_model.parameters(), lr=learning_rate
                )
                print(f"Created noise_model in epoch {epoch}")
        else:
            # continue hybrid training
            assert (
                noise_model is not None and noise_optim is not None
            ), "failed to create noise model"

            model.train()
            noise_model.train()
            train_res = train_hybrid_nad(
                model_type,
                model,
                noise_model,
                train_loader,
                optim,
                noise_optim,
                criterion,
                device,
                scaler=scaler,
                beta=beta,
            )

            model.eval()
            with torch.no_grad():
                val_loss, val_acc, val_auc = train_or_eval_base(
                    model,
                    model_type,
                    val_loader,
                    criterion,
                    device,
                    scaler=scaler,
                    nad=True,
                )

            # save metrics
            save_metrics_nad(
                fold,
                epoch,
                train_res[0],  # hybrid_loss
                train_res[1],  # model_loss
                train_res[2],  # noise_model_loss
                train_res[3],  # train_acc
                train_res[4],  # train_auc
                val_loss,
                val_acc,
                val_auc,
                path=logging_path,
            )

            s = (
                f"Hybrid-Training [Fold {fold:2d}, Epoch {epoch:2d}, Hy-Loss: {train_res[0]:8.6f}, "
                f"Model-Loss: {train_res[1]:8.6f}, Noise-Model-Loss: {train_res[2]:8.6f}, "
                f"Acc: {train_res[3]:.4f}, AUC: {train_res[4]:.4f}"
            )
            print(s)
            print(
                f"Evaluation: [Fold {fold:2d}, Epoch {epoch:2d}, Loss: {val_loss:8.6f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}]"
            )

            reg_auc = regularized_auc(train_res[4], val_auc, threshold=0)
            num_overfitted = num_overfitted + 1 if not reg_auc else 0
            num_neg_progress = num_neg_progress + 1 if val_auc < highest_val_auc else 0
            if num_overfitted == early_stop or num_neg_progress == early_stop:
                print(f"Early stopped after epoch {epoch}.")
                print(
                    f"num_overfitted={num_overfitted}, num_neg_progress={num_neg_progress}"
                )
            _path = param_path + f"auc{reg_auc:.4f}_fold{fold}_epoch{epoch}.pt"
            if fold == 1 and epoch == 1:
                # save first model in all cases
                torch.save(model.state_dict(), _path)
            elif reg_auc > highest_val_auc:
                highest_val_auc = reg_auc
                torch.save(model.state_dict(), _path)

    return highest_val_auc


def train_or_eval_base(
    model, model_type, loader, criterion, device, optim=None, scaler=None, nad=None
):
    epoch_loss, num_correct, total = 0, 0, 0
    preds, lbls = [], []

    if model_type == "T5":
        assert scaler is not None, "fp16 scaler needed for T5 training."
        for seq, att, lbl in tqdm(
            loader,
            desc="Train: " if optim is not None else "Eval: ",
            file=sys.stdout,
            unit="batches",
        ):
            seq, att, lbl = seq.to(device), att.to(device), lbl.to(device)

            with torch.cuda.amp.autocast():  # type: ignore
                logits = model(seq, att)
                loss = criterion(logits, lbl)

            if optim is not None:
                optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

            if nad is not True:
                num_correct += ((logits > 0) == lbl).sum().item()
                preds.extend(logits.detach().tolist())
            else:
                pred = logits.argmax(dim=1)
                num_correct += (pred == lbl).sum().item()
                preds.extend(logits[:, 1].detach().tolist())
            epoch_loss += loss.item()
            total += lbl.shape[0]
            lbls.extend(lbl.detach().tolist())

    elif model_type == "Padded":
        for seq, lbl, lengths in tqdm(
            loader,
            desc="Train: " if optim is not None else "Eval: ",
            file=sys.stdout,
            unit="batches",
        ):
            seq, lbl = seq.to(device), lbl.to(device)

            logits = model(seq, lengths)
            loss = criterion(logits, lbl)

            if optim is not None:
                optim.zero_grad()
                loss.backward()
                optim.step()

            if nad is not True:
                num_correct += ((logits > 0) == lbl).sum().item()
                preds.extend(logits.detach().tolist())
            else:
                pred = logits.argmax(dim=1)
                num_correct += (pred == lbl).sum().item()
                preds.extend(logits[:, 1].detach().tolist())
            epoch_loss += loss.item()
            total += lbl.shape[0]
            lbls.extend(lbl.detach().tolist())

    elif model_type == "FwBw":
        for bw_seq, fw_seq, lbl in tqdm(
            loader,
            desc="Train: " if optim is not None else "Eval: ",
            file=sys.stdout,
            unit="batches",
        ):
            bw_seq, fw_seq, lbl = bw_seq.to(device), fw_seq.to(device), lbl.to(device)

            logits = model(bw_seq, fw_seq)
            loss = criterion(logits, lbl)

            if optim is not None:
                optim.zero_grad()
                loss.backward()
                optim.step()

            if nad is not True:
                num_correct += ((logits > 0) == lbl).sum().item()
                preds.extend(logits.detach().tolist())
            else:
                pred = logits.argmax(dim=1)
                num_correct += (pred == lbl).sum().item()
                preds.extend(logits[:, 1].detach().tolist())
            epoch_loss += loss.item()
            total += lbl.shape[0]
            lbls.extend(lbl.detach().tolist())

    else:
        for seq, lbl in tqdm(
            loader,
            desc="Train: " if optim is not None else "Eval: ",
            file=sys.stdout,
            unit="batches",
        ):
            seq, lbl = seq.to(device), lbl.to(device)

            logits = model(seq)
            loss = criterion(logits, lbl)

            if optim is not None:
                optim.zero_grad()
                loss.backward()
                optim.step()

            if nad is not True:
                num_correct += ((logits > 0) == lbl).sum().item()
                preds.extend(logits.detach().tolist())
            else:
                pred = logits.argmax(dim=1)
                num_correct += (pred == lbl).sum().item()
                preds.extend(logits[:, 1].detach().tolist())
            epoch_loss += loss.item()
            total += lbl.shape[0]
            lbls.extend(lbl.detach().tolist())
    return epoch_loss / total, num_correct / total, roc_auc_score(lbls, preds)


@torch.no_grad()
def get_confusion_matrix(model, model_type, loader, device, scaler=None):
    preds, lbls = [], []

    if model_type == "T5":
        assert scaler is not None, "fp16 scaler needed for T5 training."
        for seq, att, lbl in tqdm(
            loader,
            desc="Conf Matrix Run",
            file=sys.stdout,
            unit="batches",
        ):
            seq, att, lbl = seq.to(device), att.to(device), lbl.to(device)

            with torch.cuda.amp.autocast():  # type: ignore
                logits = model(seq, att)

            pred = logits.argmax(dim=1)
            preds.extend(pred.detach().tolist())
            lbls.extend(lbl.detach().tolist())

    elif model_type == "Padded":
        for seq, lbl, lengths in tqdm(
            loader,
            desc="Conf Matrix Run",
            file=sys.stdout,
            unit="batches",
        ):
            seq, lbl = seq.to(device), lbl.to(device)
            logits = model(seq, lengths)

            pred = logits.argmax(dim=1)
            preds.extend(pred.detach().tolist())
            lbls.extend(lbl.detach().tolist())

    elif model_type == "FwBw":
        for bw_seq, fw_seq, lbl in tqdm(
            loader,
            desc="Conf Matrix Run",
            file=sys.stdout,
            unit="batches",
        ):
            bw_seq, fw_seq, lbl = bw_seq.to(device), fw_seq.to(device), lbl.to(device)
            logits = model(bw_seq, fw_seq)

            pred = logits.argmax(dim=1)
            preds.extend(pred.detach().tolist())
            lbls.extend(lbl.detach().tolist())

    else:
        for seq, lbl in tqdm(
            loader,
            desc="Conf Matrix Run",
            file=sys.stdout,
            unit="batches",
        ):
            seq, lbl = seq.to(device), lbl.to(device)
            logits = model(seq)

            pred = logits.argmax(dim=1)
            preds.extend(pred.detach().tolist())
            lbls.extend(lbl.detach().tolist())
    return torch.from_numpy(confusion_matrix(lbls, preds))


def train_hybrid_nad(
    model_type,
    model,
    noise_model,
    loader,
    optim,
    noise_optim,
    criterion,
    device,
    scaler=None,
    beta=0.8,
):
    epoch_loss, model_loss, noise_loss, num_correct, total = 0, 0, 0, 0, 0
    preds, lbls = [], []

    if model_type == "T5":
        assert scaler is not None, "fp16 scaler needed for T5 training."
        for seq, att, lbl in tqdm(
            loader,
            desc="Train: " if optim is not None else "Eval: ",
            file=sys.stdout,
            unit="batches",
        ):
            seq, att, lbl = seq.to(device), att.to(device), lbl.to(device)

            with torch.cuda.amp.autocast():  # type: ignore
                logits = model(seq, att)
                noise_logits = noise_model(logits)

                loss = criterion(logits, lbl)
                _noise_loss = criterion(noise_logits, lbl)

                weighted_loss = beta * _noise_loss + (1 - beta) * loss

            optim.zero_grad()
            noise_optim.zero_grad()
            scaler.scale(weighted_loss).backward()
            scaler.step(optim)
            scaler.step(noise_optim)
            scaler.update()

            epoch_loss += weighted_loss.item()
            model_loss += loss.item()
            noise_loss += _noise_loss.item()
            num_correct += (noise_logits.argmax(dim=1) == lbl).sum().item()
            total += lbl.shape[0]
            preds.extend(noise_logits[:, 1].detach().tolist())
            lbls.extend(lbl.detach().tolist())

    elif model_type == "Padded":
        for seq, lbl, lengths in tqdm(
            loader,
            desc="Train: " if optim is not None else "Eval: ",
            file=sys.stdout,
            unit="batches",
        ):
            seq, lbl = seq.to(device), lbl.to(device)

            logits = model(seq, lengths)
            noise_logits = noise_model(logits)

            loss = criterion(logits, lbl)
            _noise_loss = criterion(noise_logits, lbl)

            weighted_loss = beta * _noise_loss + (1 - beta) * loss

            optim.zero_grad()
            noise_optim.zero_grad()
            weighted_loss.backward()
            optim.step()
            noise_optim.step()

            epoch_loss += weighted_loss.item()
            model_loss += loss.item()
            noise_loss += _noise_loss.item()
            num_correct += (noise_logits.argmax(dim=1) == lbl).sum().item()
            total += lbl.shape[0]
            preds.extend(noise_logits[:, 1].detach().tolist())
            lbls.extend(lbl.detach().tolist())

    elif model_type == "FwBw":
        for bw_seq, fw_seq, lbl in tqdm(
            loader,
            desc="Train: " if optim is not None else "Eval: ",
            file=sys.stdout,
            unit="batches",
        ):
            bw_seq, fw_seq, lbl = bw_seq.to(device), fw_seq.to(device), lbl.to(device)

            logits = model(bw_seq, fw_seq)
            noise_logits = noise_model(logits)

            loss = criterion(logits, lbl)
            _noise_loss = criterion(noise_logits, lbl)

            weighted_loss = beta * _noise_loss + (1 - beta) * loss

            optim.zero_grad()
            noise_optim.zero_grad()
            weighted_loss.backward()
            optim.step()
            noise_optim.step()

            epoch_loss += weighted_loss.item()
            model_loss += loss.item()
            noise_loss += _noise_loss.item()
            num_correct += (noise_logits.argmax(dim=1) == lbl).sum().item()
            total += lbl.shape[0]
            preds.extend(noise_logits[:, 1].detach().tolist())
            lbls.extend(lbl.detach().tolist())

    else:
        for seq, lbl in tqdm(
            loader,
            desc="Train: " if optim is not None else "Eval: ",
            file=sys.stdout,
            unit="batches",
        ):
            seq, lbl = seq.to(device), lbl.to(device)

            logits = model(seq)
            noise_logits = noise_model(logits)

            loss = criterion(logits, lbl)
            _noise_loss = criterion(noise_logits, lbl)

            weighted_loss = beta * _noise_loss + (1 - beta) * loss

            optim.zero_grad()
            noise_optim.zero_grad()
            weighted_loss.backward()
            optim.step()
            noise_optim.step()

            epoch_loss += weighted_loss.item()
            model_loss += loss.item()
            noise_loss += _noise_loss.item()
            num_correct += (noise_logits.argmax(dim=1) == lbl).sum().item()
            total += lbl.shape[0]
            preds.extend(noise_logits[:, 1].detach().tolist())
            lbls.extend(lbl.detach().tolist())

    return (
        epoch_loss / total,
        model_loss / total,
        noise_loss / total,
        num_correct / total,
        roc_auc_score(lbls, preds),
    )


def train_or_eval_coteaching(
    model_type,
    loader,
    model1,
    model2,
    device,
    forget_rate=None,
    cot_criterion=None,
    criterion=None,
    optim1=None,
    optim2=None,
    scaler1=None,
    scaler2=None,
    cot_plus_train=None,
):
    epoch_loss1, epoch_loss2, num_correct1, num_correct2, total = 0, 0, 0, 0, 0
    preds1, preds2, lbls = [], [], []

    if model_type == "T5":
        assert (
            scaler1 is not None and scaler2 is not None
        ), "two fp16 scalers needed for coteaching T5 training."
        for seq, att, lbl in tqdm(
            loader,
            desc="Train: " if optim1 is not None else "Eval: ",
            file=sys.stdout,
            unit="batches",
        ):
            seq, att, lbl = seq.to(device), att.to(device), lbl.to(device)

            with torch.cuda.amp.autocast():  # type: ignore
                logits1 = model1(seq, att)
                pred1 = logits1 > 0
                logits2 = model2(seq, att)
                pred2 = logits2 > 0

            if cot_plus_train is not None:
                # co-teaching-plus training
                assert (
                    cot_criterion is not None
                    and criterion is not None
                    and forget_rate is not None
                ), "need both criterions and forget_rate defined"

                idx = torch.where(pred1 != pred2)[0]
                with torch.cuda.amp.autocast():  # type: ignore
                    if idx.shape[0] * (1 - forget_rate) < 0:
                        loss1 = criterion(logits1, lbl)
                        loss2 = criterion(logits2, lbl)
                    else:
                        loss1, loss2 = cot_criterion(logits1, logits2, lbl, forget_rate)
            elif cot_plus_train is None and criterion is None:
                # co-teaching training
                assert (
                    cot_criterion is not None and forget_rate is not None
                ), "cot_criterion and forget_rate needed for coteaching training"

                with torch.cuda.amp.autocast():  # type: ignore
                    loss1, loss2 = cot_criterion(logits1, logits2, lbl, forget_rate)
            else:
                # co-teaching and co-teaching-plus evaluation
                assert (
                    cot_plus_train is None and cot_criterion is None
                ), "only criterion needed for evaluation"
                assert criterion is not None, "criterion needed for evaluation"

                with torch.cuda.amp.autocast():  # type: ignore
                    loss1 = criterion(logits1, lbl)
                    loss2 = criterion(logits2, lbl)

            if optim1 is not None and optim2 is not None:
                optim1.zero_grad()
                scaler1.scale(loss1).backward()
                scaler1.step(optim1)
                scaler1.update()

                optim2.zero_grad()
                scaler2.scale(loss2).backward()
                scaler2.step(optim2)
                scaler2.update()

            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()
            num_correct1 += (pred1 == lbl).sum().item()
            num_correct2 += (pred2 == lbl).sum().item()
            total += lbl.shape[0]
            preds1.extend(logits1.detach().tolist())
            preds2.extend(logits2.detach().tolist())
            lbls.extend(lbl.detach().tolist())

    elif model_type == "Padded":
        for seq, lbl, lengths in tqdm(
            loader,
            desc="Train: " if optim1 is not None else "Eval: ",
            file=sys.stdout,
            unit="batches",
        ):
            seq, lbl = seq.to(device), lbl.to(device)

            logits1 = model1(seq, lengths)
            pred1 = logits1 > 0
            logits2 = model2(seq, lengths)
            pred2 = logits2 > 0

            if cot_plus_train is not None:
                # co-teaching-plus training
                assert (
                    cot_criterion is not None
                    and criterion is not None
                    and forget_rate is not None
                ), "need both criterions and forget_rate defined"

                idx = torch.where(pred1 != pred2)[0]
                if idx.shape[0] * (1 - forget_rate) < 0:
                    loss1 = criterion(logits1, lbl)
                    loss2 = criterion(logits2, lbl)
                else:
                    loss1, loss2 = cot_criterion(logits1, logits2, lbl, forget_rate)
            elif cot_plus_train is None and criterion is None:
                # co-teaching training
                assert (
                    cot_criterion is not None and forget_rate is not None
                ), "cot_criterion and forget_rate needed for coteaching training"

                loss1, loss2 = cot_criterion(logits1, logits2, lbl, forget_rate)
            else:
                # co-teaching and co-teaching-plus evaluation
                assert (
                    cot_plus_train is None and cot_criterion is None
                ), "only criterion needed for evaluation"
                assert criterion is not None, "criterion needed for evaluation"
                loss1 = criterion(logits1, lbl)
                loss2 = criterion(logits2, lbl)

            if optim1 is not None and optim2 is not None:
                optim1.zero_grad()
                loss1.backward()
                optim1.step()

                optim2.zero_grad()
                loss2.backward()
                optim2.step()

            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()
            num_correct1 += (pred1 == lbl).sum().item()
            num_correct2 += (pred2 == lbl).sum().item()
            total += lbl.shape[0]
            preds1.extend(logits1.detach().tolist())
            preds2.extend(logits2.detach().tolist())
            lbls.extend(lbl.detach().tolist())

    elif model_type == "FwBw":
        for bw_seq, fw_seq, lbl in tqdm(
            loader,
            desc="Train: " if optim1 is not None else "Eval: ",
            file=sys.stdout,
            unit="batches",
        ):
            bw_seq, fw_seq, lbl = bw_seq.to(device), fw_seq.to(device), lbl.to(device)

            logits1 = model1(bw_seq, fw_seq)
            pred1 = logits1 > 0
            logits2 = model2(bw_seq, fw_seq)
            pred2 = logits2 > 0

            if cot_plus_train is not None:
                # co-teaching-plus training
                assert (
                    cot_criterion is not None
                    and criterion is not None
                    and forget_rate is not None
                ), "need both criterions and forget_rate defined"

                idx = torch.where(pred1 != pred2)[0]
                if idx.shape[0] * (1 - forget_rate) < 0:
                    loss1 = criterion(logits1, lbl)
                    loss2 = criterion(logits2, lbl)
                else:
                    loss1, loss2 = cot_criterion(logits1, logits2, lbl, forget_rate)
            elif cot_plus_train is None and criterion is None:
                # co-teaching training
                assert (
                    cot_criterion is not None and forget_rate is not None
                ), "cot_criterion and forget_rate needed for coteaching training"

                loss1, loss2 = cot_criterion(logits1, logits2, lbl, forget_rate)
            else:
                # co-teaching and co-teaching-plus evaluation
                assert (
                    cot_plus_train is None and cot_criterion is None
                ), "only criterion needed for evaluation"
                assert criterion is not None, "criterion needed for evaluation"
                loss1 = criterion(logits1, lbl)
                loss2 = criterion(logits2, lbl)

            if optim1 is not None and optim2 is not None:
                optim1.zero_grad()
                loss1.backward()
                optim1.step()

                optim2.zero_grad()
                loss2.backward()
                optim2.step()

            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()
            num_correct1 += (pred1 == lbl).sum().item()
            num_correct2 += (pred2 == lbl).sum().item()
            total += lbl.shape[0]
            preds1.extend(logits1.detach().tolist())
            preds2.extend(logits2.detach().tolist())
            lbls.extend(lbl.detach().tolist())

    else:
        for seq, lbl in tqdm(
            loader,
            desc="Train: " if optim1 is not None else "Eval: ",
            file=sys.stdout,
            unit="batches",
        ):
            seq, lbl = seq.to(device), lbl.to(device)

            logits1 = model1(seq)
            pred1 = logits1 > 0
            logits2 = model2(seq)
            pred2 = logits2 > 0

            if cot_plus_train is not None:
                # co-teaching-plus training
                assert (
                    cot_criterion is not None
                    and criterion is not None
                    and forget_rate is not None
                ), "need both criterions and forget rate defined"

                idx = torch.where(pred1 != pred2)[0]
                if idx.shape[0] * (1 - forget_rate) < 0:
                    loss1 = criterion(logits1, lbl)
                    loss2 = criterion(logits2, lbl)
                else:
                    loss1, loss2 = cot_criterion(logits1, logits2, lbl, forget_rate)
            elif cot_plus_train is None and criterion is None:
                # co-teaching training
                assert (
                    cot_criterion is not None and forget_rate is not None
                ), "cot_criterion and forget_rate needed for coteaching training"

                loss1, loss2 = cot_criterion(logits1, logits2, lbl, forget_rate)
            else:
                # co-teaching and co-teaching-plus evaluation
                assert (
                    cot_plus_train is None and cot_criterion is None
                ), "only criterion needed for evaluation"
                assert criterion is not None, "criterion needed for evaluation"
                loss1 = criterion(logits1, lbl)
                loss2 = criterion(logits2, lbl)

            if optim1 is not None and optim2 is not None:
                optim1.zero_grad()
                loss1.backward()
                optim1.step()

                optim2.zero_grad()
                loss2.backward()
                optim2.step()

            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()
            num_correct1 += (pred1 == lbl).sum().item()
            num_correct2 += (pred2 == lbl).sum().item()
            total += lbl.shape[0]
            preds1.extend(logits1.detach().tolist())
            preds2.extend(logits2.detach().tolist())
            lbls.extend(lbl.detach().tolist())

    return (
        epoch_loss1 / total,
        epoch_loss2 / total,
        num_correct1 / total,
        num_correct2 / total,
        roc_auc_score(lbls, preds1),
        roc_auc_score(lbls, preds2),
    )


def train_or_eval_jocor(
    model_type,
    loader,
    model1,
    model2,
    device,
    forget_rate=None,
    jocor_criterion=None,
    optim=None,
    scaler=None,
):
    epoch_loss, num_correct1, num_correct2, total = 0, 0, 0, 0
    preds1, preds2, lbls = [], [], []

    if model_type == "T5":
        assert scaler is not None, "fp16 scaler needed for T5."
        for seq, att, lbl in tqdm(
            loader,
            desc="Train: " if optim is not None else "Eval: ",
            file=sys.stdout,
            unit="batches",
        ):
            seq, att, lbl = seq.to(device), att.to(device), lbl.to(device)

            with torch.cuda.amp.autocast():  # type: ignore
                logits1 = model1(seq, att)
                pred1 = logits1 > 0
                logits2 = model2(seq, att)
                pred2 = logits2 > 0

                if jocor_criterion is not None:
                    loss = jocor_criterion(logits1, logits2, lbl, forget_rate)
                else:
                    loss = torch.tensor(-1)

            if optim is not None:
                optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

            epoch_loss += loss.item()
            num_correct1 += (pred1 == lbl).sum().item()
            num_correct2 += (pred2 == lbl).sum().item()
            total += lbl.shape[0]
            preds1.extend(logits1.detach().tolist())
            preds2.extend(logits2.detach().tolist())
            lbls.extend(lbl.detach().tolist())

    elif model_type == "Padded":
        for seq, lbl, lengths in tqdm(
            loader,
            desc="Train: " if optim is not None else "Eval: ",
            file=sys.stdout,
            unit="batches",
        ):
            seq, lbl = seq.to(device), lbl.to(device)

            logits1 = model1(seq, lengths)
            pred1 = logits1 > 0
            logits2 = model2(seq, lengths)
            pred2 = logits2 > 0

            if optim is not None and jocor_criterion is not None:
                loss = jocor_criterion(logits1, logits2, lbl, forget_rate)

                optim.zero_grad()
                loss.backward()
                optim.step()
            else:
                loss = torch.tensor(-1)

            epoch_loss += loss.item()
            num_correct1 += (pred1 == lbl).sum().item()
            num_correct2 += (pred2 == lbl).sum().item()
            total += lbl.shape[0]
            preds1.extend(logits1.detach().tolist())
            preds2.extend(logits2.detach().tolist())
            lbls.extend(lbl.detach().tolist())

    elif model_type == "FwBw":
        for bw_seq, fw_seq, lbl in tqdm(
            loader,
            desc="Train: " if optim is not None else "Eval: ",
            file=sys.stdout,
            unit="batches",
        ):
            bw_seq, fw_seq, lbl = bw_seq.to(device), fw_seq.to(device), lbl.to(device)

            logits1 = model1(bw_seq, fw_seq)
            pred1 = logits1 > 0
            logits2 = model2(bw_seq, fw_seq)
            pred2 = logits2 > 0

            if optim is not None and jocor_criterion is not None:
                loss = jocor_criterion(logits1, logits2, lbl, forget_rate)

                optim.zero_grad()
                loss.backward()
                optim.step()
            else:
                loss = torch.tensor(-1)

            epoch_loss += loss.item()
            num_correct1 += (pred1 == lbl).sum().item()
            num_correct2 += (pred2 == lbl).sum().item()
            total += lbl.shape[0]
            preds1.extend(logits1.detach().tolist())
            preds2.extend(logits2.detach().tolist())
            lbls.extend(lbl.detach().tolist())

    else:
        for seq, lbl in tqdm(
            loader,
            desc="Train: " if optim is not None else "Eval: ",
            file=sys.stdout,
            unit="batches",
        ):
            seq, lbl = seq.to(device), lbl.to(device)

            logits1 = model1(seq)
            pred1 = logits1 > 0
            logits2 = model2(seq)
            pred2 = logits2 > 0

            if optim is not None and jocor_criterion is not None:
                loss = jocor_criterion(logits1, logits2, lbl, forget_rate)

                optim.zero_grad()
                loss.backward()
                optim.step()
            else:
                loss = torch.tensor(-1)

            epoch_loss += loss.item()
            num_correct1 += (pred1 == lbl).sum().item()
            num_correct2 += (pred2 == lbl).sum().item()
            total += lbl.shape[0]
            preds1.extend(logits1.detach().tolist())
            preds2.extend(logits2.detach().tolist())
            lbls.extend(lbl.detach().tolist())

    return (
        epoch_loss / total,
        num_correct1 / total,
        num_correct2 / total,
        roc_auc_score(lbls, preds1),
        roc_auc_score(lbls, preds2),
    )

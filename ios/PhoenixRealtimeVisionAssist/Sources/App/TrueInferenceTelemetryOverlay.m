#import <UIKit/UIKit.h>
#import <notify.h>

static NSString * const LiteViewTelemetryNotificationName = @"com.phoenix.realtimevisionassist.broadcast.true-inference.v1";
static NSString * const LiteViewFrameDiagnosticsNotificationName = @"com.phoenix.realtimevisionassist.broadcast.frame-diagnostics.v1";
static NSInteger const LiteViewTelemetryOverlayTag = 84015;
static int LiteViewTelemetryToken = -1;
static int LiteViewFrameDiagnosticsToken = -1;

@interface LiteViewTrueInferenceTelemetryOverlay : NSObject
@end

@implementation LiteViewTrueInferenceTelemetryOverlay

+ (void)load {
    dispatch_async(dispatch_get_main_queue(), ^{
        notify_register_check(LiteViewTelemetryNotificationName.UTF8String, &LiteViewTelemetryToken);
        notify_register_check(LiteViewFrameDiagnosticsNotificationName.UTF8String, &LiteViewFrameDiagnosticsToken);
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(sceneDidActivate:)
                                                     name:UISceneDidActivateNotification
                                                   object:nil];
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(sceneDidActivate:)
                                                     name:UIApplicationDidBecomeActiveNotification
                                                   object:nil];
        [NSTimer scheduledTimerWithTimeInterval:0.5
                                         target:self
                                       selector:@selector(refreshTimer:)
                                       userInfo:nil
                                        repeats:YES];
        [self installOrRefresh];
    });
}

+ (void)sceneDidActivate:(NSNotification *)notification {
    [self installOrRefresh];
}

+ (void)refreshTimer:(NSTimer *)timer {
    if (UIApplication.sharedApplication.applicationState == UIApplicationStateActive) {
        [self installOrRefresh];
    }
}

+ (UIWindow *)activeWindow {
    for (UIScene *scene in UIApplication.sharedApplication.connectedScenes) {
        if (![scene isKindOfClass:UIWindowScene.class]) { continue; }
        UIWindowScene *windowScene = (UIWindowScene *)scene;
        if (windowScene.activationState != UISceneActivationStateForegroundActive &&
            windowScene.activationState != UISceneActivationStateForegroundInactive) {
            continue;
        }
        for (UIWindow *window in windowScene.windows) {
            if (window.isKeyWindow) { return window; }
        }
        if (windowScene.windows.firstObject) { return windowScene.windows.firstObject; }
    }
    return nil;
}

+ (void)installOrRefresh {
    UIWindow *window = [self activeWindow];
    if (!window) { return; }

    UIView *panel = [window viewWithTag:LiteViewTelemetryOverlayTag];
    UILabel *label = nil;
    if (!panel) {
        panel = [[UIView alloc] initWithFrame:CGRectZero];
        panel.tag = LiteViewTelemetryOverlayTag;
        panel.translatesAutoresizingMaskIntoConstraints = NO;
        panel.backgroundColor = [UIColor colorWithWhite:0.04 alpha:0.88];
        panel.layer.cornerRadius = 14.0;
        panel.layer.masksToBounds = YES;
        panel.userInteractionEnabled = NO;

        label = [[UILabel alloc] initWithFrame:CGRectZero];
        label.tag = LiteViewTelemetryOverlayTag + 1;
        label.translatesAutoresizingMaskIntoConstraints = NO;
        label.numberOfLines = 0;
        label.font = [UIFont monospacedSystemFontOfSize:11.0 weight:UIFontWeightSemibold];
        label.textColor = UIColor.whiteColor;
        label.adjustsFontSizeToFitWidth = YES;
        label.minimumScaleFactor = 0.72;
        [panel addSubview:label];
        [window addSubview:panel];

        UILayoutGuide *safe = window.safeAreaLayoutGuide;
        [NSLayoutConstraint activateConstraints:@[
            [panel.leadingAnchor constraintEqualToAnchor:safe.leadingAnchor constant:12.0],
            [panel.trailingAnchor constraintEqualToAnchor:safe.trailingAnchor constant:-12.0],
            [panel.bottomAnchor constraintEqualToAnchor:safe.bottomAnchor constant:-10.0],
            [label.leadingAnchor constraintEqualToAnchor:panel.leadingAnchor constant:12.0],
            [label.trailingAnchor constraintEqualToAnchor:panel.trailingAnchor constant:-12.0],
            [label.topAnchor constraintEqualToAnchor:panel.topAnchor constant:9.0],
            [label.bottomAnchor constraintEqualToAnchor:panel.bottomAnchor constant:-9.0]
        ]];
    } else {
        label = (UILabel *)[panel viewWithTag:LiteViewTelemetryOverlayTag + 1];
    }

    if (!label) { return; }
    label.text = [self telemetryText];
    [window bringSubviewToFront:panel];
}

+ (NSString *)telemetryText {
    uint64_t inferenceState = 0;
    BOOL inferenceReady = LiteViewTelemetryToken >= 0 &&
        notify_get_state(LiteViewTelemetryToken, &inferenceState) == NOTIFY_STATUS_OK &&
        (inferenceState & (UINT64_C(1) << 63)) != 0;

    uint64_t frameState = 0;
    BOOL frameReady = LiteViewFrameDiagnosticsToken >= 0 &&
        notify_get_state(LiteViewFrameDiagnosticsToken, &frameState) == NOTIFY_STATUS_OK &&
        (frameState & (UINT64_C(1) << 63)) != 0;

    if (!inferenceReady && !frameReady) {
        return @"候选真推理遥测 · 等待 ReplayKit 首次 detector 扫描";
    }

    uint64_t coreML = inferenceReady ? (inferenceState & UINT64_C(0x0FFF)) : 0;
    uint64_t decoded = inferenceReady ? ((inferenceState >> 12) & UINT64_C(0x0FFF)) : 0;
    uint64_t nonEmpty = inferenceReady ? ((inferenceState >> 24) & UINT64_C(0x0FFF)) : 0;
    uint64_t failures = inferenceReady ? ((inferenceState >> 42) & UINT64_C(0x003F)) : 0;
    uint64_t model = inferenceReady ? ((inferenceState >> 48) & UINT64_C(0x0003)) : 0;
    uint64_t decoder = inferenceReady ? ((inferenceState >> 50) & UINT64_C(0x0007)) : 0;
    uint64_t source = inferenceReady ? ((inferenceState >> 53) & UINT64_C(0x0003)) : 0;

    if (!frameReady) {
        return [NSString stringWithFormat:
                @"候选真推理遥测 · 384px Direct Core ML\n"
                 "Core ML %llu · 解码 %llu · 非空目标 %llu · 失败 %llu\n"
                 "模型 %@ · 解码 %@ · 来源 %@\n"
                 "帧链诊断：等待首次 detector 扫描",
                coreML,
                decoded,
                nonEmpty,
                failures,
                [self modelName:model],
                [self decoderName:decoder],
                [self sourceName:source]];
    }

    uint64_t preprocessSuccesses = frameState & UINT64_C(0x0FFF);
    uint64_t preprocessFailures = (frameState >> 12) & UINT64_C(0x0FFF);
    uint64_t pixelFormat = (frameState >> 24) & UINT64_C(0x0003);
    uint64_t orientation = (frameState >> 26) & UINT64_C(0x000F);
    BOOL preprocessOK = ((frameState >> 30) & UINT64_C(1)) != 0;
    BOOL latestCoreML = ((frameState >> 31) & UINT64_C(1)) != 0;
    BOOL latestDecode = ((frameState >> 32) & UINT64_C(1)) != 0;
    BOOL latestNonEmpty = ((frameState >> 33) & UINT64_C(1)) != 0;
    BOOL latestInferenceFailed = ((frameState >> 34) & UINT64_C(1)) != 0;
    uint64_t detectorSequence = (frameState >> 35) & UINT64_C(0x00FF);

    return [NSString stringWithFormat:
            @"候选真推理遥测 · 384px Direct Core ML\n"
             "Core ML %llu · 解码 %llu · 非空目标 %llu · 失败 %llu\n"
             "输入 %@ · 方向 %@ · 预处理 %llu/%llu · detector seq %llu\n"
             "最近链：预处理%@ · ML%@ · 解码%@ · 输出%@ · 失败%@\n"
             "模型 %@ · 解码 %@ · 来源 %@",
            coreML,
            decoded,
            nonEmpty,
            failures,
            [self pixelFormatName:pixelFormat],
            [self orientationName:orientation],
            preprocessSuccesses,
            preprocessFailures,
            detectorSequence,
            preprocessOK ? @"✓" : @"×",
            latestCoreML ? @"✓" : @"×",
            latestDecode ? @"✓" : @"×",
            latestNonEmpty ? @"有" : @"空",
            latestInferenceFailed ? @"是" : @"否",
            [self modelName:model],
            [self decoderName:decoder],
            [self sourceName:source]];
}

+ (NSString *)modelName:(uint64_t)code {
    switch (code) {
        case 1: return @"yolo11n";
        case 2: return @"旧YOLOv3Tiny";
        case 3: return @"其他 Core ML";
        default: return @"尚未加载";
    }
}

+ (NSString *)decoderName:(uint64_t)code {
    switch (code) {
        case 1: return @"RecognizedObject";
        case 2: return @"coordinates+confidence";
        case 3: return @"Ultralytics raw";
        case 4: return @"empty output";
        case 5: return @"unsupported";
        default: return @"未确定";
    }
}

+ (NSString *)sourceName:(uint64_t)code {
    switch (code) {
        case 1: return @"Core ML";
        case 2: return @"Vision Tracker";
        case 3: return @"旧Vision fallback";
        default: return @"无目标/待确认";
    }
}

+ (NSString *)pixelFormatName:(uint64_t)code {
    switch (code) {
        case 1: return @"BGRA";
        case 2: return @"NV12 420f";
        case 3: return @"NV12 420v";
        default: return @"未知格式";
    }
}

+ (NSString *)orientationName:(uint64_t)code {
    switch (code) {
        case 1: return @"up(1)";
        case 2: return @"upMirrored(2)";
        case 3: return @"down(3)";
        case 4: return @"downMirrored(4)";
        case 5: return @"leftMirrored(5)";
        case 6: return @"right(6)";
        case 7: return @"rightMirrored(7)";
        case 8: return @"left(8)";
        default: return [NSString stringWithFormat:@"unknown(%llu)", code];
    }
}

@end
